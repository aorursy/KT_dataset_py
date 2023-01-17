import time 
import lxml.etree as ET
import pandas as pd

file_path = '/kaggle/input/french-reddit-discussion/final_SPF_2.xml'

#Initializes the parser
parser = ET.XMLParser(recover=True)
#Parses the file
tree = ET.parse(file_path, parser=parser)
xroot = tree.getroot()


#Prepares our final df
dfcols = ['link_id', 'subreddit_id', 'uid',"comment_id",'score', 'parent_id', 'create_utc', 'text']
df_xml = pd.DataFrame(columns=dfcols)

#Processes the file
nb_of_text = 10**4 #If you want more or less conversations you can change it here. If you want everything just put len(xroot) 
start = time.time()
s = start
i = 0
#Iterates overs the topics
for node in xroot[0:nb_of_text]:
    if (i+1) % 1000 == 0:
        e = time.time()
        print(str(i + 1) + ' iterations done in ' + str(round(e - s)) + " sec.")
        s = time.time()
    link_id = node.attrib.get('link_id')
    subreddit_id = node.attrib.get('subreddit_id')
    #Iterates over the posts into the conv
    for j in range(len(node.getchildren())):
        uid = node.getchildren()[j].get('uid')
        comment_id = node.getchildren()[j].get('comment_id')
        score = node.getchildren()[j].get('score')
        parent_id = node.getchildren()[j].get('parent_id')
        create_utc = node.getchildren()[j].get('create_utc')
        text = node.getchildren()[j].text
        df_xml = df_xml.append(pd.Series([link_id, subreddit_id,
                                          uid,comment_id,score,parent_id,create_utc,text],
                                         index=dfcols)
                               ,ignore_index=True)
    i+=1
        
end = time.time()
print("Finished. Done in : " +  str(round(end - start)) + 'seconds')

#df_xml.to_csv('Reddit_Conv_french.csv')
final = pd.read_csv('/kaggle/input/reddit-conv-french/Reddit_Conv_final_french.csv',index_col = 0)
final.head()
print(final.columns)
print(final.shape)
final.to_csv('Reddit_Conv_french.csv')