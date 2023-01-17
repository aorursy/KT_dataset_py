import pandas as pd
EdgesGraph = pd.read_csv("../input/french-twitter-graph-csv-complete-all-edges/clean_complete.csv", names = ['id', 'friend', 'sequence'])
import gc
EdgesGraphSmaller = EdgesGraph[['id', 'friend']]
del EdgesGraph
gc.collect()
#EdgesGraphSmaller["id"]= EdgesGraphSmaller["id"].astype(str)
nodeDetails = pd.read_csv('../input/detailsonfrenchtwitterpopulation/allnodesdetails.csv')
import igraph as ig
#counter = 0
records = EdgesGraphSmaller.to_records(index=False)
#for x in EdgesGraphSmaller.values :
#    print(x[1])
#    counter = counter + 1
#    if counter > 5:
#        break
#tuples = [tuple(x) for x in EdgesGraphSmaller.values]

#EdgesGraphSmaller = EdgesGraph[['id', 'friend']]
#testGrph = ig.Graph.TupleList(EdgesGraphSmaller.itertuples(index=False), directed=False, weights=False)

del EdgesGraphSmaller
gc.collect()
records[0]
gc.collect()
records = records[:int(len(records)*1/4)]
len(records)
import sys
sys.getsizeof(list(records))
testGrph = ig.Graph.TupleList(records, directed=False, weights=False)
giant = testGrph.components().giant()
totalCommunities = giant.community_multilevel()
giant.modularity(totalCommunities)
len(totalCommunities)
allCommunities = totalCommunities.subgraphs()
for i in allCommunities[0].vs:
    print(i['name'])
    break
whichCommunity = 5
print(len(totalCommunities[whichCommunity]))
test = []
for i, j in zip(allCommunities[whichCommunity].vs.degree(), allCommunities[whichCommunity].vs):
    
    if i >= 100 or j['name'] == 80820758 or j['name'] == 801822323825410049:
        try:
            name = nodeDetails[nodeDetails['id'] == j['name']]['screen_name'].values[0]
            test.append((j['name'], i, name))
        except:
            pass
print(sorted(test, key=lambda x: x[1]))
annotated = pd.read_csv('../input/dataset-for-french/profilesmanualannotation.csv')
annotated.head()
psSet = set(annotated.loc[annotated['party'] == 'ps']['UserId'])
fnSet = set(annotated.loc[annotated['party'] == 'fn']['UserId'])
lrSet = set(annotated.loc[annotated['party'] == 'lr']['UserId'])
fiSet = set(annotated.loc[annotated['party'] == 'fi']['UserId'])
emSet = set(annotated.loc[annotated['party'] == 'em']['UserId'])
emPSSet = set(annotated.loc[annotated['party'] == 'em/ps']['UserId'])
fnLRSet = set(annotated.loc[annotated['party'] == 'fn/lr']['UserId'])
emFISet = set(annotated.loc[annotated['party'] == 'em/fi']['UserId'])
emLRSet = set(annotated.loc[annotated['party'] == 'em/lr']['UserId'])
fiPSSet = set(annotated.loc[annotated['party'] == 'fi/ps']['UserId'])
fiFNSet = set(annotated.loc[annotated['party'] == 'fi/fn']['UserId'])
fiLRSet = set(annotated.loc[annotated['party'] == 'fi/lr']['UserId'])
fnPSSet = set(annotated.loc[annotated['party'] == 'fn/ps']['UserId'])
lrPSSet = set(annotated.loc[annotated['party'] == 'lr/ps']['UserId'])
emFNSet = set(annotated.loc[annotated['party'] == 'em/fn']['UserId'])
import random
#allCommunities = totalCommunities.subgraphs()
communityCount = []
#dataFrameWithCommunities = pd.DataFrame(communityCount, columns = ['ps', 'lr', 'fi', 'em', 'fn'])
counter = 1
for i in allCommunities:
    addendum = [0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
    Community = counter
    for j in i.vs:
        number = random.randrange(0, 100, 1)
        myName = j['name']
        if myName in psSet:
            addendum[0] = addendum[0] + 1
        elif myName in lrSet:
            addendum[1] = addendum[1] + 1
        elif myName in fiSet:
            addendum[2] = addendum[2] + 1
        elif myName in emSet:
            addendum[3] = addendum[3] + 1
        elif myName in fnSet:
            addendum[4] = addendum[4] + 1
        elif myName in emPSSet:
            addendum[5] = addendum[5] + 1
        elif myName in fnLRSet:
            addendum[6] = addendum[6] + 1
        elif myName in emFISet:
            addendum[7] = addendum[7] + 1
        elif myName in emLRSet:
            addendum[8] = addendum[8] + 1
        elif myName in fiPSSet:
            addendum[9] = addendum[9] + 1
        elif myName in fiFNSet:
            addendum[10] = addendum[10] + 1
        elif myName in fiLRSet:
            addendum[11] = addendum[11] + 1
        elif myName in fnPSSet:
            addendum[12] = addendum[12] + 1
        elif myName in lrPSSet:
            addendum[13] = addendum[13] + 1
        elif myName in emFNSet:
            addendum[14] = addendum[14] + 1
    addendum.append(i.vcount())
    communityCount.append(addendum)
        #if (number == 55 or number == 56) and annotated.loc[annotated['UserId'] == j['name']]['party'].size > 0:
        #    temp = [j['name'], annotated.loc[annotated['UserId'] == j['name']]['party'].values[0],Community]
        #    print(temp)
             
    counter = counter + 1
dataFrameWithCommunities = pd.DataFrame(communityCount, columns = ['ps', 'lr', 'fi', 'em', 'fn', 'em/ps', 'fn/lr', 'em/fi',
       'em/lr', 'fi/ps', 'fi/fn', 'fi/lr', 'fn/ps', 'lr/ps', 'em/fn', 'total'])
dataFrameWithCommunities