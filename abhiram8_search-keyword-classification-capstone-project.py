import os 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
#os.chdir("../input/ipba-capston-data")
os.listdir()
data=pd.read_csv('keywords.csv')
data.head(2)
data.text.nunique()
a=pd.Series(data["text"].value_counts().sort_values(ascending=False).head(30))
a.plot(kind="bar")
import nltk
import re
from nltk.corpus import stopwords
SW=['abacus', 'abacus class', 'academi', 'academi educ pvt ltd', 'academi situat',
    'account payabl sap', 'address', 'administ test', 'advanc iot data cluster', 'algorithm',
    'aml', 'appl inc','appliedia', 'baconni', 'bank', 'basic', 'bioinformat', 'blockjain', 
    'calend', 'cfa', 'chennai', 'compar domain', 'coursewar', 'datamit', 'domain', 'dual', 'durat univers',
    'email', 'email market','finanicail', 'forum', 'full', 'full pack', 'full sta', 'full stack', 'fullstack', 
    'general', 'iim bangalor', 'iim indor', 'imm', 'invest bank', 'invest bank program', 'jigsawplanet', 'jogsaw',
    'kafeel', 'languag', 'mac app', 'mail address founder', 'math', 'model', 'msw', 'mumbai', 'offic', 'oir test', 
    'path', 'pen test', 'plant zombi garden warfar', 'prereq', 'program', 'pyhtion', 'referr', 'respons model', 'retail', 'sale',
    'sale manag', 'sasdasd', 'servic', 'short', 'specialist', 'studi tour', 'summari statist', 'survey monkey link starter', 'test', 'tool', 
    'trail', 'train', 'trainer', 'unicorn', 'visual merchandis', 'visualis', 'workday hcm', 'workshop', 'freerto']
SW=SW+["cours","faculti","placement","chat","free","ultim","certif","enrol","now","pay","later","jlc","deepak","aagri","jigsaw","canva",\
                                      "jlc","journal","onlin","offlin","career","net","student","www","com","bilu","xxx","video","baigapipariya","ryt","jigsawacademi","https","talent","india","whatsapp","coursera","godiva","chocol","freelearn"\
                                     'special','bundl','retail','train','becom','take','commerc','sampl paper',"stack","advanc","specialis","use","fullstack","stat","beggin","intermedi","beginn","bachelor","leader","foundat","oozi","websit",'plant','zombi', 'garden', 'warfar',"special"\
                                     "becom","understand","cpurs",'invest',"compar","morn","routin","associ","pd","master","special","postgradu","post","graduat","microsoft","specal","courss","mail","founder",\
                                     "appl","inc","accont","workday","hcm","begin","oir","exampl","min",'situat',\
                                      'educ pvt ltd',"price",'pen','pack','sta','respons','freelearn','manag',"journey",'survey','monkey','link','starter','account','payabl','sap','visual','merchandis',\
                                      'iim','bangalor','indor','durat','univers','administ','mac','store']
len(SW)
stopwords=stopwords.words("english")+SW
len(stopwords)
#lets see how words are linked:
for line in data["text"]:
    if re.search("^visual merchandis",line):
            print(line)
#checking if the removal of a word will render many records empty
for line in data["text"]:
    if re.search("^pen",line):
            print(line)
#let's see what are the words that have common initial characters
for line in data["text"]:
    line=line.split(" ")
    for i in line:
        if re.search("^master",i):
            print(i)
def clean(text):
    line=text.strip()
    line=re.sub("\s+"," ",line)
    line=re.sub("[^\w\s]"," ",line)
    split=line.split(" ")
    cleaned=[word for word in split if word not in stopwords and len(word)>2]
    result=" ".join(cleaned)
    return result
dat1=data
dat1["new_text"]=data["text"].apply(clean)
dat1["new_text"].nunique()
new=dat1.query("new_text!=''")#removing rows that have empty text after cleaning
new.drop(["text"],inplace=True,axis=1)
new.shape #old 2712
new
#the idea is to create a nested list to work with because 1D list is not preferrable for our case
def ls(text):
    line=text.strip()
    split=line.split(" ")
    cleaned=[word for word in split if word not in stopwords and len(word)>2]
    return cleaned
cln_tex=new["new_text"].apply(ls) #
cln_tex=cln_tex.tolist()
len(cln_tex)
single=[]
double=[]
triple=[]
quadraple=[]
for i in range(0,2607):
    if len(cln_tex[i])==1:
        single.extend(cln_tex[i])
    if len(cln_tex[i])==2:
        double.extend(cln_tex[i])
    if len(cln_tex[i])==3:
        triple.extend(cln_tex[i])
    if len(cln_tex[i])==4:
        quadraple.extend(cln_tex[i])
    
    
pd.Series(single).unique()

#
pd.Series(quadraple).unique()
len(cln_tex)

for i in range(0,2607):
        if len(cln_tex[i])==1:
                if re.search("^ana",(cln_tex[i][0])):
                    cln_tex[i][0]="analytics" 
                    
                elif re.search("^datasc",(cln_tex[i][0])):
                    cln_tex[i][0]="data science"  
                    
                elif re.search("^tab",(cln_tex[i][0])):
                    cln_tex[i][0]="tableau"
                    
                elif re.search("(mlai)|(pgpmlai)|(machin)|(mllib)",(cln_tex[i][0])):
                    cln_tex[i][0]="machine learning"
                    
                elif re.search("(bigdata)|(bigd)",cln_tex[i][0]):
                    cln_tex[i][0]="big data"
                    
                elif re.search("^artifi",cln_tex[i][0]):
                    cln_tex[i][0]="artificial intelligence"
                    
                elif re.search("^bus",cln_tex[i][0]):
                    cln_tex[i][0]="business"
                    
                elif re.search("aiml",cln_tex[i][0]):
                    cln_tex[i][0]="artificial intelligence"
                
                elif re.search("^ex",cln_tex[i][0]):
                    cln_tex[i][0]="excel"
                else:
                    cln_tex[i][0]=cln_tex[i][0]
                    
        elif  len(cln_tex[i])==2:
            
            for j in range(0,2):
                
                if re.search("^ana",(cln_tex[i][j])):
                    cln_tex[i][j]="analytics" 
                    
                elif re.search("^datasc",(cln_tex[i][j])):
                    cln_tex[i][j]="data science"
                    
                elif re.search("(busi)|(buis)",(cln_tex[i][j])):
                    cln_tex[i][j]="business"
                    
                elif re.search("^sci",(cln_tex[i][j])):
                    cln_tex[i][j]="science"
                    
                elif re.search("^artifi",(cln_tex[i][j])):
                    cln_tex[i][j]="artificial"
                    
                elif re.search("^intel",(cln_tex[i][j])):
                    cln_tex[i][j]="intelligence"
                    
                elif re.search("^mach",(cln_tex[i][j])):
                    cln_tex[i][j]="machine"
                
                elif re.search("^learn",(cln_tex[i][j])):
                    cln_tex[i][j]="learning"
                 
                else:
                    cln_tex[i][j]=cln_tex[i][j]
     
                
        elif  len(cln_tex[i])==3:
            
            for k in range (0,3):
               
                if re.search("^ana",(cln_tex[i][k])):
                    cln_tex[i][k]="analytics" 
                    
                elif re.search("^sci",(cln_tex[i][k])):
                    cln_tex[i][k]="science" 
                    
                elif re.search("machin",(cln_tex[i][k])):
                    cln_tex[i][k]="machine"
                    
                elif re.search("(mlai)|(pgpmlai)",(cln_tex[i][k])):
                    cln_tex[i][k]="machine learning"
                    
                elif re.search("pyhton",(cln_tex[i][k])):
                    cln_tex[i][k]="python"
                else:
                    cln_tex[i][k]=cln_tex[i][k]  
                
                
        elif len(cln_tex[i])==4:
            
            for x in range (0,4):
                
                if re.search("^ana",(cln_tex[i][x])):
                    cln_tex[i][x]="analytics" 
                    
                elif re.search("^sci",(cln_tex[i][x])):
                    cln_tex[i][x]="science"
            
                elif re.search("^artifi",(cln_tex[i][x])):
                    cln_tex[i][x]="artificial"
                
                elif re.search("^intel",(cln_tex[i][x])):
                    cln_tex[i][x]="intelligence" 
                    
                elif re.search("^mach",(cln_tex[i][x])):
                    cln_tex[i][x]="machine"
                
                elif re.search("^learn",(cln_tex[i][x])):
                    cln_tex[i][x]="learning" 
                else:
                    cln_tex[i][x]=cln_tex[i][x]
                    
                    
                
    
stringlst=[" ".join(str(i) for i in lst) for lst in cln_tex]
new["rep"]=stringlst 
new["rep"].head() #cleaned and replaced words column
new["rep"]=new["rep"].map(lambda x: "business analytics"  if (x in ["ipa","statist","integr analyt","integr programm ","integr programm analytics","integr analytics ipa","integr analyt ipa","ipba","busi analyst","statist busi analytics",'integr analytics']) else x)
excel=["vba","excel vba","excel advanc analyt","excel macro","analyt excel","visual basic",'analytics execl']
new["rep"]=new["rep"].map(lambda x: "excel"  if (x in excel) else x)
iot=["internet thing"]
new["rep"]=new["rep"].map(lambda x: "iot"  if (x in iot) else x)
dat_sc=["time seri analysi","bootcamp","boot camp","data science python","python data science","data science bootcamp","data science boot camp","bootcamp data science","data sceinec","project manag data science"]
new["rep"]=new["rep"].map(lambda x: "data science"  if (x in dat_sc) else x)
bigdat=['bigdata','big data bootcamp','bigdata engin','big data applic enggin','bigdata administratior','machin learn pyspark big data','bigdata cloud','salari big data']

new["rep"]=new["rep"].map(lambda x: "big data"  if (x in bigdat) else x)
new["rep"].nunique()
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(ngram_range=(1,2),max_features=500,stop_words=stopwords)
tfv=tf.fit_transform(new["rep"])
mat=pd.DataFrame(tfv.toarray(),columns=tf.get_feature_names())
mat
mat.sum().sort_values(ascending=False).head(30).plot.bar(figsize=(20,5))
import sklearn.cluster as cluster
clusters=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
compactness=[]
for i in clusters:
    mod=cluster.KMeans(n_clusters=i,random_state=42,)
    mod=mod.fit(mat)
    compactness.append(mod.inertia_)
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(clusters,compactness,".")
model=cluster.KMeans(n_clusters=30,random_state=42)
model=model.fit(mat)
new["kmeans_labels"]=model.labels_
print(new[new["kmeans_labels"]==2]["rep"].unique())
import seaborn as sns
plt.figure(figsize=(10,5))
sns.countplot(x="kmeans_labels",data=new)
plt.show() 

from wordcloud import WordCloud
labelno=[0,1,2,3,4,5,6,7,8,9]
lab1=new.query('kmeans_labels=="0"')
wordstring=" ".join(lab1["rep"])
wrdcloud=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud)
    
lab2=new.query('kmeans_labels=="1"')
wordstring=" ".join(lab2["rep"])
wrdcloud1=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud1)

    
lab3=new.query('kmeans_labels=="2"')
wordstring=" ".join(lab3["rep"])
wrdcloud2=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud2)

    
lab4=new.query('kmeans_labels=="3"')
wordstring=" ".join(lab4["rep"])
wrdcloud3=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud3)
    
lab5=new.query('kmeans_labels=="4"')
wordstring=" ".join(lab5["rep"])
wrdcloud4=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud4)

    
lab6=new.query('kmeans_labels=="5"')
wordstring=" ".join(lab6["rep"])
wrdcloud5=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud5)

    
lab7=new.query('kmeans_labels=="6"')
wordstring=" ".join(lab7["rep"])
wrdcloud6=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud6)
    
lab8=new.query('kmeans_labels=="7"')
wordstring=" ".join(lab8["new_text"])
wrdcloud7=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud7)
    
lab9=new.query('kmeans_labels=="8"')
wordstring=" ".join(lab9["rep"])
wrdcloud8=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud8)

    
lab10=new.query('kmeans_labels=="9"')
wordstring=" ".join(lab10["rep"])
wrdcloud9=WordCloud(width=800,height=800,max_words=1000,background_color="black").generate_from_text(wordstring)
plt.figure(figsize=[10,10])
plt.imshow(wrdcloud9)    
    
    
bus=['business analytics','business','mis analytics','macro','collect analytics',\
    'adob analytics','analytics lab','qlick','qliksens','qlik','qlickview',\
     'qlikview','spss','linear regress','case studi analytics','roulett data','business partner',\
    'busi analytics manag',"saap",'probabl',"business intelligence",'time seri analytics',\
    'analytics work','introduct analytics analytics','visual analytics','analytics trial'\
    'introduct analytics','cluster analytics','stati','anlyt','powerbi','storytel data',\
    'business accumen','business acumen','msbi','time seri','analytics expert','storytel',\
    'analytics sas','busi intellig analytics','analytics research','time','busi case studi','sampl paper ibpa',\
    'sampl paper ipba','analytics bfsi','analytics stori tell','stori tell analytics','stori','stori tell data',\
 'stori tell','analytics']

dat=['python','data science','data science sas','tableau','data analytics','data analytics sas',\
    'data analytics python','data data analytics','data analytics ysis','data analytics fresher'\
    'ibm watson','ibm','ibm iib','matlab','data','python django','predict analytics'\
    'python data science','categor data analytics','categor data','latent variabl','data pre',\
    'etl','data adv','data science fee','data visual','watson',\
    "data science fee structur","data py",'data engin','numpi','develop science','data scientist sas python sas',\
    'data analytics fee','data science old','pgpdms','data anyt','data anslyt','data mine','phyton',\
    'data smart engin','data smart','data visual programm','data smart begin','pgpdm data science',\
    'data science architect','time data science','applic pgpdm','cloudera','data science excel',\
    'cost data analytics','clean data','phython','data ledger']

big=['big data analytics','hadoop','hadoop spark','hadoop admin','spark','apach','apach spark',\
    'big data','big data hadoop','big data engin','big','data science big data','kafka',\
    'elk','mapreduc','big data develop','trial class big data','vmware','apach cassandra',\
    'pyspark',"cassandra",'teradata','apach','big data analyst data scienc',"julia",'cloudlab',\
    'storm','scala spark','spark scala','big data architect','data warehous','wiley big data',\
    'data wareh','big data administr','spark stream','cloud lab','flume','scala','scal']

ml=['machine learning','data science machine learning','data analytics machine learning',\
   'deep learning','deep','text mine','text','nlp','cluster','tensorflow','neural network',\
   'sentiment analytics','decis tree',"uipath",'datasci machine learn','robot autom','natur'\
   'machine learning decison tree','deep learn python',"decis tree",'support vector machine',
   'network','natur process','data scienc machin learn time','machine learn pyspark',\
   'deeplearn','random forest','neural network python','machine learn engin'\
   'supervis learning','supervis','neural','tensor flow','bag boost','text minin','random',\
   'random forest neural network','data structur machine learning','machine learn python','artificial intelligence',\
  'machine learning artificial intelligence','rpa',"deep neural network"]

excel=['power','power pivot','excel','analytics excel','excel analytics','visual',"busi statist excel"]

iot=['iot','iot fee','iot webinar','iot profession','iot analytics',"iot salari report",'iot bootcamp','iot data cluster',\
    'layer iot','fee iot']

other=['sap','sap hana logist','sap hana','sap logist','netapp','shell script','perl'\
      'javascript','android','android develop','informatica','mongodb','angular','databas',\
      'mainfram','web analytics',"informatica mdm",'softwar',"css",'devapp','webinar','webinar data analytics',\
      'data set','oper manag','women analytics','python webdevelop' 'webdevelop' 'web develop',\
      'suppli chain','oper analytics','solut architect','devlop','devop','java','java devolop',"mongodb","mongobd",'salari analytics'
      ,'sas','base sas','clinic sas','sas base','sas python','sas fee','googl analytics']

blockchain=["blockchain","bitcoin",'blockchain technolog',"block",'block chain technolog','block chain']

HR=['peopl analytics','consult','digit market','market analytics','market','social media analytics',\
   'peopl analytics organ','ecommerc analytics']

support=['login','batch','crash','payment','renew access','trial','recommend system',\
        'script','feedback','support','recours',"dashboard",'refund','lab','user dashboard'\
        'salari report',"job",'resum build','fee','mail founder','class',\
         'studi tour',"classroom","class","studi tour",'fee structur','grade','corpor login','classroom',\
        'learning center','learning centr','syllabus','regist','learn','support team','faq','social login',\
        'job report','report','log','corpor','applic','resum alreadi upload portal','offer','mobil app','app',
         'app develop', 'app download']
labels=[]
for i in new["rep"]:
    if i in bus:
        labels.append("business analytics")
    elif i in dat:
        labels.append("data science")
    elif i in big:
        labels.append("big data")
    elif i in ml:
        labels.append("MLAI")
    elif i in other:
        labels.append("other")
    elif i in excel:
        labels.append("excel")
    elif i in iot:
        labels.append("IOT")
    elif i in blockchain:
        labels.append("blockchain")
    elif i in HR:
        labels.append("HR analytics")
    else:
        labels.append("support")
                
new["labels"]=labels
new.dtypes
new["date"]=pd.to_datetime(new["date"])
new
X=mat
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
categories=['HR analytics', 'IOT', 'MLAI', 'big data', 'blockchain',
       'business analytics', 'data science', 'excel', 'other', 'support']
enc.fit(categories)
y=enc.fit_transform(new["labels"])
y
enc.classes_
new["label_encode"]=y
print(y)
new[new["label_encode"]==9]["labels"].unique()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=42)
X_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
np.floor(np.sqrt(len(mat.columns.tolist()))) #square root of number of features in DTM
import sklearn
list(range(1,6))
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
#accuracy_score(Y_test,classifier.predict(X_test))
parameters=[{"max_depth":[None],"n_estimators":[30,50,70,90,110],"max_features":[2,3,5,11,13,15,17,21],"random_state":[42]}]

grid_search=GridSearchCV(estimator=RandomForestClassifier(),param_grid=parameters,scoring="accuracy",n_jobs=-1)

grid_search=grid_search.fit(X_train,Y_train)
grid_search.best_score_
grid_search.best_params_
rf=RandomForestClassifier(n_estimators=90,max_features=11,max_depth=None,criterion="gini",n_jobs=-1,random_state=42)
mod_RF=rf.fit(X_train,Y_train)
feature=pd.Series(mod_RF.feature_importances_,index=X_train.columns.tolist()).sort_values(ascending=False).head(30)
feature
feature.index
plt.figure(figsize=(10,10))
sns.barplot(x=feature,y=feature.index)
y_test_pred=mod_RF.predict(X_test)
RFtest=accuracy_score(Y_test,y_test_pred) 
RFtrain=accuracy_score(Y_train,mod_RF.predict(X_train))
RFtest
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
mod_log=logreg.fit(X_train,Y_train)
preds=mod_log.predict(X_test)
logtest=accuracy_score(Y_test,preds)
logtrain=accuracy_score(Y_train,mod_log.predict(X_train))
print("training accuracy is",logtrain)
print("whereas test accuracy is",logtest)
pd.crosstab(Y_test,preds)
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
mod_nb=nb.fit(X_train,Y_train)
nbpreds=mod_nb.predict(X_test)
nbtest=accuracy_score(Y_test,nbpreds)
nbtrain=accuracy_score(Y_train,mod_nb.predict(X_train))
from sklearn.svm import SVC
SVC=SVC(kernel="linear")
modsvc=SVC.fit(X=X_train,y=Y_train)
svcpreds=modsvc.predict(X_test)
svctest=accuracy_score(Y_test,svcpreds)
svctrain=accuracy_score(Y_train,modsvc.predict(X_train))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ABC=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=42)
mod_ABC=ABC.fit(X_train,Y_train)
ACpreds=mod_ABC.predict(X_test)
adatest=accuracy_score(Y_test,ACpreds)
adatrain=accuracy_score(Y_train,mod_ABC.predict(X_train))
from xgboost import XGBClassifier
XGBClassifier()
xgb=XGBClassifier(max_depth=10,n_estimators=200,n_jobs=-1,booster="gblinear",learning_rate=0.1,min_child_weight=0.5,random_state=42)
modx=xgb.fit(X_train,Y_train)
predboost=modx.predict(X_test)
boosttest=accuracy_score(Y_test,predboost)
boosttrain=accuracy_score(Y_train,modx.predict(X_train))
diction={"model":["Random Forest","Logistic Regression","Naive Bayes","SVC","Adaboost Classifier","XGBoost"],\
     "test_accuracy":[RFtest,logtest,nbtest,svctest,adatest,boosttest],\
      "train_accuracy":[RFtrain,logtrain,nbtrain,svctrain,adatrain,boosttrain]}
pd.DataFrame.from_dict(diction)
new.sort_values(by="date",ascending=False,inplace=True)
new.dtypes
courses=new["labels"].tolist()
from nltk.probability import FreqDist
freq1=FreqDist(courses)
freq1.plot(10,title="COURSES")
plt.show()
new[new["label_encode"]==2]
interval_30=(new["date"]>='2019-11-05 13:56:00')&(new["date"]<='2019-12-05 13:56:00')
interval_90=(new["date"]>='2019-9-07 13:56:00')&(new["date"]<='2019-12-05 13:56:00')
monthdat=new.loc[interval_30]
quarterdat=new.loc[interval_90]
freq2=FreqDist(monthdat["labels"].tolist())
freq2.plot(10,title="COURSES")
plt.show()
freq3=FreqDist(quarterdat["labels"].tolist())
freq3.plot(10,title="COURSES")
plt.show()
txt=["natural language processing","analytics with excel","data science with python"]
txt
dfme=pd.DataFrame(tf.transform(txt).toarray(),columns=tf.get_feature_names())
dfme.shape
modsvc.predict(dfme)
mod_log.predict(dfme)
textdata=new["rep"].sample(n=100)
type(textdata)
txtdf=textdata.to_frame()
classtxt=pd.DataFrame(tf.transform(textdata).toarray(),columns=tf.get_feature_names())
classtxt.shape
txtdf["predlab"]=enc.inverse_transform(modsvc.predict(classtxt))
txtdf.head(15)



