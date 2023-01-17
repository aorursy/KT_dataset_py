import pandas as pd

import numpy as np
# def load_chat(sender,reciever,isgroup,path="WhatsApp Chat with All PRO ABL,MP ,INS,FCI.txt"):

#     file = open(path,errors='ignore')

#     lines=file.readlines()

#     for i in range(len(lines)):

#         try:

#             if not lines[i].strip():

# #                 print("empty line",i)

#                 lines[i-1]=lines[i-1]+' ';

#                 lines.pop(i)

#             elif(lines[i][0] not in ["0","1","2","3"]):

#                 lines[i-1]=lines[i-1]+' '+lines[i]

#                 lines.pop(i)

#         except:

#             continue

# #             print(lines[i-4])

#     #         print(i-1,lines[i])

#     #     print(lines[i-1])

# #     file1.writelines(lines)

#     s_list=[]

#     r_list=[]

#     g_list=[]

#     ts_list=[]

#     body_list=[]

#     date_l=[]

#     month_l=[]

#     year_l=[]

#     time_l=[]

#     for l in lines:

# #         print(l)

#         if not l.strip():

#             continue

#         try:

#             ts,body=l.split('-',1)

#         except:

#             print(l)

#         try:

#             s,body= body.split(':',1)

#             s=s.strip()

#             body=body.strip()

#         except:

#             continue

# #         print(s==sender)

#         dmy,time=ts.split(',')

#         time=time.strip()

#         date,month,year=dmy.split('/')

#         if (s==sender):

#             r_list.append(None if isgroup else reciever)

#         else:

#             r_list.append(None if isgroup else s)

# #         print(s,sender,b,(s,reciever)[b])

#         g_list.append(reciever if isgroup else None)

#         s_list.append(s)

# #         ts_list.append(ts)

#         date_l.append(date)

#         month_l.append(month)

#         year_l.append(year)

#         time_l.append(time)

#         body_list.append(body)

#     d=pd.DataFrame({"Sender_name":s_list,"Receiver_name":r_list,"Decrypted Raw Message ":body_list,"Year":year_l,"Month":month_l,"Day":date_l,"Time":time_l,"Forwarded":[None]*len(s_list),"Group Name":g_list,"Group Info":[None]*len(s_list)})

#     return d



# # execute the following lines only if sure 

# a=load_chat("Shubham Gogate","All PRO ABL,MP ,INS,FCI",True,"WhatsApp Chat with All PRO ABL,MP ,INS,FCI.txt")

# d=pd.read_csv('o.csv')

# a.to_csv('o.csv')
!pip3 install newspaper3k

import newspaper
toi_paper = newspaper.build("https://timesofindia.indiatimes.com/india",memoize_articles=False)

for article in toi_paper.articles:

    print(article.url)
from newspaper import Article
for category in toi_paper.category_urls():

     print(category)
for feed_url in toi_paper.feed_urls():

    print(feed_url)
first_article = toi_paper.articles[3]

first_article = Article(first_article.url,language='en')
first_article.download()
first_article.parse()
print(first_article.text)
first_article.nlp()
print(first_article.summary)
print(first_article.keywords)
newspaper.hot()
newspaper.popular_urls()
cat = toi_paper.category_urls()
# for category in toi_paper.category_urls():

print(cat[0])

toi_cat_paper = newspaper.build(cat[0],language='en',memoize_articles=False)

# print(toi_cat_paper)

for article in toi_cat_paper.articles:

        print(article.keywords)
news_dict = {'Category':[],'News_Url':[],'Title':[],'Article':[],'Summary':[],'Keywords':[]}
toi_cat_paper = newspaper.build(cat[0],language='en',memoize_articles=False)

# for article in toi_cat_paper.articles:

article = toi_cat_paper.articles[0]

print(article)
for article in toi_cat_paper.articles:

    try:

        article.download()

        article.parse()

        news_dict['Category'].append(cat[0])

        news_dict['News_Url'].append(article.url)

        news_dict['Title'].append(article.title)

        news_dict['Article'].append(article.text)

        article.nlp()

        news_dict['Summary'].append(article.summary)

        news_dict['Keywords'].append(article.keywords)

    except Exception as e:

        print(e)
pd.DataFrame.from_dict(news_dict)
pd.DataFrame.from_dict(news_dict)