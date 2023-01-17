import os
import sys
print(sys.version)
print('\nContents of input:\n %s' % os.listdir('../input'))
print('\nContents of snappy:\n %s' % os.listdir('../input/snappy'))
print('\nContents of snap-memetracker-raw:\n %s' % os.listdir('../input/snap-memetracker-raw'))
print('\nContents of the working directory:\n %s' % os.listdir('.'))
! cp ../input/snappy/snap.py snap.py
! cp ../input/snappy/_snap.so _snap.so
! cp ../input/snappy/setup.py setup.py
! pwd
! ls -l
! python setup.py install
import snap
import pandas as pd
go = ''
variant = ['']
container = []

for line in open('../input/snap-memetracker-raw/clust-qt08080902w3mfq5.txt/clust-qt08080902w3mfq5.txt').readlines():
    if not (line.startswith('\t\t') or line.startswith('\t') or line=='\n'):
        # find the header line
        if 'you can put lipstick on a pig' in line:
            go = 'on'
        else:
            go = 'off'
    # get quotes of the phrase
    if go == 'on':
        # get variant
        if line.startswith('\t') and not line.startswith('\t\t'):
            variant[0] = line.split('\t')[3]
        # get quote
        if line.startswith('\t\t'):
            container.append(variant + line.strip().split('\t'))

df = pd.DataFrame(container, columns=['Variant','Date', 'Fq', 'Type', 'URL'])

df['Date'] = pd.to_datetime(df['Date'])
df.info()
df.sample(n=10)
i = 0
# SNAP hash tables to store URL to NodeId mapping
urlNode = snap.TStrIntH()
nodeUrl = snap.TIntStrH()

for item in df['URL'].unique():
    urlNode[item] = i
    nodeUrl[i] = item
    i += 1
nodeUrl[42]
urlNode['http://methlabhomes.com/2008/11/foreclosure-lipstick-tricks-that-you-should-know']
urlNode.IsKey('http://methlabhomes.com/2008/11/foreclosure-lipstick-tricks-that-you-should-know')
df['NodeId'] = df['URL'].apply(lambda x: urlNode[x])
%%time

G1 = snap.TNGraph.New()

for page in df['URL'].unique():
    G1.AddNode(urlNode[page])
# iterators for:
i = 0 # lines - total
j = 0 # pages of interest 
k = 0 # outgoing links
l = 0 # links to other pages of interest

container = []

source = ''
moment = ''
copyLinks = 'off'

for line in open('../input/snap-memetracker-raw/quotes_2008-09.txt/quotes_2008-09.txt').readlines():
    lineSp = line.strip().split('\t')
    
    if lineSp[0] == 'P':
        if urlNode.IsKey(lineSp[1]):
            source = lineSp[1]
            copyLinks = 'on'
            j += 1
            
    if lineSp[0] == 'T':
        moment = lineSp[1]
    
    if line == '\n':
        copyLinks = 'off'

    if copyLinks == 'on':
        if lineSp[0] == 'L':
            k += 1
            if urlNode.IsKey(lineSp[1]):
                try:
                    G1.AddEdge(urlNode[source], urlNode[lineSp[1]])
                    container.append([urlNode[source], moment, urlNode[lineSp[1]]])
                except:
                    pass
                l += 1

    i += 1
print('''
Total lines processed: \t\t\t %d
Pages of interest: \t\t\t %d
Outgoing links: \t\t\t %d
Links to other pages of interest: \t %d
''' % (i,j,k,l))
dfL = pd.DataFrame(container, columns=['Source','Date', 'Target'])
dfL.info()
dfL.sample(n=10)
snap.PrintInfo(G1, 'Who points to whom', 'output.txt', False)
for line in open('output.txt').readlines():
    print (line.strip())
# Suprematist comment #1
#snap.DelZeroDegNodes(G1)
snap.PrintInfo(G1, 'Who points to whom - trimmed disconnected nodes', 'output.txt', False)
for line in open('output.txt').readlines():
    print (line.strip())
from IPython.display import Image

NIdColorH = snap.TIntStr64H()

for Node in G1.Nodes():
    NIdColorH[Node.GetId()] = 'black'

snap.DrawGViz(G1, snap.gvlNeato, 'f001.png', 'Study 1. Who points to whom', False, NIdColorH )
Image(filename='f001.png') 
# Suprematist comments #2 and #3
#snap.DelSelfEdges(G1)
#snap.DelZeroDegNodes(G1)

snap.DrawGViz(G1, snap.gvlNeato, 'f002.png', 'Study 2. Who points to whom - The White Square')
Image(filename='f002.png') 
snap.PrintInfo(G1, 'Who points to whom - trimmed disconnected nodes, removed self-edges', 'output.txt', False)
for line in open('output.txt').readlines():
    print (line.strip())
MxWcc = snap.GetMxWcc(G1)
MxScc = snap.GetMxScc(G1)

for Node in G1.Nodes():
    if MxWcc.IsNode(Node.GetId()):
        NIdColorH[Node.GetId()] = 'white'
        if MxScc.IsNode(Node.GetId()):
            NIdColorH[Node.GetId()] = 'red'
    else:
        NIdColorH[Node.GetId()] = 'black'
        
snap.DrawGViz(G1,
              snap.gvlNeato, 
              'f003.png', 
              'Study 3. Who points to whom - highlighted maximum connected components\n'
              '(red for strong, white for weak ties)',
              False,
              NIdColorH)

Image(filename='f003.png') 
df['Color'] = df['Type'].map({'B': 'black', 'M': 'yellow'})
df.sample(n=10)
for item in df[['NodeId', 'Color']].values:
    NIdColorH[item[0]] = item[1]

snap.DrawGViz(G1, snap.gvlNeato, 'f004.png', 'Study 4. Who points to whom - highlighted mainstream media',False, NIdColorH)
Image(filename='f004.png') 
PRankH = snap.TIntFlt64H()

snap.GetPageRank(G1, PRankH)

PRankH.SortByDat(False)

l = 0
for item in PRankH:
    if l < 10:
        print (nodeUrl[item], PRankH[item])
        l += 1
        NIdColorH[item] = 'purple'
    else:
        NIdColorH[item] = NIdColorH[item]
snap.DrawGViz(G1, snap.gvlNeato, 'f005.png', 'Study 5. Who points to whom - highlighted mainstream media, '
              'and top 10 identified by Page Rank',False, NIdColorH)
Image(filename='f005.png')
df.to_csv('nodes.csv')
dfL.to_csv('links.csv')

FOut = snap.TFOut('g1.graph')
G1.Save(FOut)
FOut.Flush()

FOut = snap.TFOut('NIdColorH.hash')
NIdColorH.Save(FOut)
FOut.Flush()