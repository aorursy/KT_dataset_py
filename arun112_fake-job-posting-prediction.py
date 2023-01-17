import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer 
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from scipy.sparse import hstack,csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
sns.set()
df=pd.read_csv("../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv")
df.head()
df.info()
df["fraudulent"].value_counts()
df["title"].nunique()
split_location=df["location"].apply(lambda x:str(x).strip().split(','))
split_location=split_location.apply(pd.Series)
split_location
split_location[~(split_location[4].isnull())]
df['country']=split_location[0]
df['location_count']=df["location"].apply(lambda x:str(x).split(', '))
df['location_count']=df['location_count'].apply(lambda x:max(len(x)-2,0))
plt.figure(figsize=(12,5))
ax=sns.countplot(x='location_count',data=df,hue='fraudulent')
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50))
df[((df['telecommuting']==1)&(df['location']!='none'))]['job_id'].count()
df[((df['telecommuting']==0)&(df['location'].isnull()))]['job_id'].count()
df['null_count']=df.isnull().sum(axis=1)
sns.countplot(x='null_count',data=df,hue='fraudulent');
# Map alpha 2 to corresponding alpha 3 country code
country_code_mapping = {"BD": "BGD", "BE": "BEL", "BF": "BFA", "BG": "BGR", "BA": "BIH", "BB": "BRB", "WF": "WLF", "BL": "BLM", "BM": "BMU", "BN": "BRN", "BO": "BOL", "BH": "BHR", "BI": "BDI", "BJ": "BEN", "BT": "BTN", "JM": "JAM", "BV": "BVT", "BW": "BWA", "WS": "WSM", "BQ": "BES", "BR": "BRA", "BS": "BHS", "JE": "JEY", "BY": "BLR", "BZ": "BLZ", "RU": "RUS", "RW": "RWA", "RS": "SRB", "TL": "TLS", "RE": "REU", "TM": "TKM", "TJ": "TJK", "RO": "ROU", "TK": "TKL", "GW": "GNB", "GU": "GUM", "GT": "GTM", "GS": "SGS", "GR": "GRC", "GQ": "GNQ", "GP": "GLP", "JP": "JPN", "GY": "GUY", "GG": "GGY", "GF": "GUF", "GE": "GEO", "GD": "GRD", "GB": "GBR", "GA": "GAB", "SV": "SLV", "GN": "GIN", "GM": "GMB", "GL": "GRL", "GI": "GIB", "GH": "GHA", "OM": "OMN", "TN": "TUN", "JO": "JOR", "HR": "HRV", "HT": "HTI", "HU": "HUN", "HK": "HKG", "HN": "HND", "HM": "HMD", "VE": "VEN", "PR": "PRI", "PS": "PSE", "PW": "PLW", "PT": "PRT", "SJ": "SJM", "PY": "PRY", "IQ": "IRQ", "PA": "PAN", "PF": "PYF", "PG": "PNG", "PE": "PER", "PK": "PAK", "PH": "PHL", "PN": "PCN", "PL": "POL", "PM": "SPM", "ZM": "ZMB", "EH": "ESH", "EE": "EST", "EG": "EGY", "ZA": "ZAF", "EC": "ECU", "IT": "ITA", "VN": "VNM", "SB": "SLB", "ET": "ETH", "SO": "SOM", "ZW": "ZWE", "SA": "SAU", "ES": "ESP", "ER": "ERI", "ME": "MNE", "MD": "MDA", "MG": "MDG", "MF": "MAF", "MA": "MAR", "MC": "MCO", "UZ": "UZB", "MM": "MMR", "ML": "MLI", "MO": "MAC", "MN": "MNG", "MH": "MHL", "MK": "MKD", "MU": "MUS", "MT": "MLT", "MW": "MWI", "MV": "MDV", "MQ": "MTQ", "MP": "MNP", "MS": "MSR", "MR": "MRT", "IM": "IMN", "UG": "UGA", "TZ": "TZA", "MY": "MYS", "MX": "MEX", "IL": "ISR", "FR": "FRA", "IO": "IOT", "SH": "SHN", "FI": "FIN", "FJ": "FJI", "FK": "FLK", "FM": "FSM", "FO": "FRO", "NI": "NIC", "NL": "NLD", "NO": "NOR", "NA": "NAM", "VU": "VUT", "NC": "NCL", "NE": "NER", "NF": "NFK", "NG": "NGA", "NZ": "NZL", "NP": "NPL", "NR": "NRU", "NU": "NIU", "CK": "COK", "XK": "XKX", "CI": "CIV", "CH": "CHE", "CO": "COL", "CN": "CHN", "CM": "CMR", "CL": "CHL", "CC": "CCK", "CA": "CAN", "CG": "COG", "CF": "CAF", "CD": "COD", "CZ": "CZE", "CY": "CYP", "CX": "CXR", "CR": "CRI", "CW": "CUW", "CV": "CPV", "CU": "CUB", "SZ": "SWZ", "SY": "SYR", "SX": "SXM", "KG": "KGZ", "KE": "KEN", "SS": "SSD", "SR": "SUR", "KI": "KIR", "KH": "KHM", "KN": "KNA", "KM": "COM", "ST": "STP", "SK": "SVK", "KR": "KOR", "SI": "SVN", "KP": "PRK", "KW": "KWT", "SN": "SEN", "SM": "SMR", "SL": "SLE", "SC": "SYC", "KZ": "KAZ", "KY": "CYM", "SG": "SGP", "SE": "SWE", "SD": "SDN", "DO": "DOM", "DM": "DMA", "DJ": "DJI", "DK": "DNK", "VG": "VGB", "DE": "DEU", "YE": "YEM", "DZ": "DZA", "US": "USA", "UY": "URY", "YT": "MYT", "UM": "UMI", "LB": "LBN", "LC": "LCA", "LA": "LAO", "TV": "TUV", "TW": "TWN", "TT": "TTO", "TR": "TUR", "LK": "LKA", "LI": "LIE", "LV": "LVA", "TO": "TON", "LT": "LTU", "LU": "LUX", "LR": "LBR", "LS": "LSO", "TH": "THA", "TF": "ATF", "TG": "TGO", "TD": "TCD", "TC": "TCA", "LY": "LBY", "VA": "VAT", "VC": "VCT", "AE": "ARE", "AD": "AND", "AG": "ATG", "AF": "AFG", "AI": "AIA", "VI": "VIR", "IS": "ISL", "IR": "IRN", "AM": "ARM", "AL": "ALB", "AO": "AGO", "AQ": "ATA", "AS": "ASM", "AR": "ARG", "AU": "AUS", "AT": "AUT", "AW": "ABW", "IN": "IND", "AX": "ALA", "AZ": "AZE", "IE": "IRL", "ID": "IDN", "UA": "UKR", "QA": "QAT", "MZ": "MOZ"}
df['country']=df['country'].apply(lambda x:country_code_mapping[x] if x!='nan' else 'nan')
fig = go.Figure(data=go.Choropleth(
    locations = df['country'].value_counts().index,
    z = df['country'].value_counts().values,
    text = df['country'].value_counts().index,
    colorscale = 'Blues',
    autocolorscale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'job ads',
))

fig.update_layout(
    title_text='Total job ads',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ),

)

fig.show()
fraud_list=df[df['fraudulent']==1]['country'].value_counts().to_dict()
total_count_list=df['country'].value_counts().to_dict()
percent_fraud_dict={}
for country in total_count_list.keys():
    if country in fraud_list:
        percent_fraud_dict[country]=fraud_list[country]/total_count_list[country]*100
    else:
        percent_fraud_dict[country]=0
percent_fraud_dict=OrderedDict(sorted(percent_fraud_dict.items())) 
fig = go.Figure(data=go.Choropleth(
    locations = list(percent_fraud_dict.keys()),
    z = list(percent_fraud_dict.values()),
    text =  list(percent_fraud_dict.keys()),
    colorscale = 'Reds',
    autocolorscale=False,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Job ads percent',
))

fig.update_layout(
    title_text='Percentage of fraudulent job ads',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ),

)

fig.show()
df['department'].value_counts().head(20)
df['department'].fillna('none',inplace=True)
sns.countplot(x='department',data=df[((df['department']=='tech')|(df['department']=='Technology'))],hue='fraudulent');
sns.countplot(x='department',data=df[((df['department']=='IT')|(df['department']=='Information Technology'))],hue='fraudulent');
sns.countplot(x='department',data=df[((df['department']=='Development')|(df['department']=='Product')|(df['department']=='Product Development'))],hue='fraudulent');
sns.countplot(x='department',data=df[df['department']=='none'],hue='fraudulent');
df['salary_range'].nunique()
df[df['salary_range'].notnull()]['salary_range']
salary=df['salary_range'].apply(lambda x:str(x).strip().split('-'))
salary=salary.apply(pd.Series)
# Function to check fot non integer values
def int_check(x):
    try:
        int(x)
    except:
        if x!='nan':
            print(x)
salary[0].apply(int_check)
def int_convert(x):
    try:
        return int(x)
    except:
        return 0
df['salary_lower_bound']=salary[0]
df['salary_lower_bound']=df['salary_lower_bound'].apply(int_convert)
df['salary_upper_bound']=salary[1]
df['salary_upper_bound']=df['salary_upper_bound'].apply(int_convert)
df['avg_salary']=(df['salary_upper_bound']+df['salary_lower_bound'])/2
px.histogram(data_frame=df[(df['avg_salary']<=100000)&(df['avg_salary']!=0)],x="avg_salary",color='fraudulent',marginal='rug')
sns.countplot(x='telecommuting',data=df,hue='fraudulent');
sns.barplot(x="has_company_logo", y="percentage", hue="fraudulent", data=df.groupby(['fraudulent'])['has_company_logo']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('has_company_logo'));
sns.barplot(x="has_questions", y="percentage", hue="fraudulent", data=df.groupby(['fraudulent'])['has_questions']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('has_questions'));
df['employment_type'].unique()
df['employment_type'].fillna("none",inplace=True)
sns.countplot(x='employment_type',data=df);
df['required_experience'].unique()
df[df['required_experience']=='Not Applicable'].head()
df[df['required_experience'].isnull()].head()
df['required_experience'].fillna('Not Applicable',inplace=True)
plt.figure(figsize=(10,5))
sns.countplot(x='required_experience',data=df);
df['required_education'].unique()
df['required_education'].fillna('Unspecified',inplace=True)
plt.figure(figsize=(10,5))
sns.countplot(y='required_education',data=df,hue='fraudulent');
df['industry'].unique()
df['industry'].fillna('not specified',inplace=True)
df['function'].unique()
df['function'].fillna('not specified',inplace=True)
df[(df['company_profile'].isnull())&(df['fraudulent']==0)]
abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
}

stop_words=stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def text_preprocess(x):
    if pd.isnull(x) or x=='nan':
        return "not specified"
    x=str(x)
    x=x.lower()
    #remove url,email,phone number and links
    x=re.sub("(#url_.*#)|(#email.*#)|(#phone.*#)|((http|https)://\w*)",' ',x)
    # remove abbreviations
    pattern = re.compile(r'\b(' + '|'.join(abbr_dict.keys()) + r')\b')
    x = pattern.sub(lambda x: abbr_dict[x.group()], x)
    tokens = re.findall("[\w']+", x)
    tokens=[word for word in tokens if ((word not in stop_words) and (len(word)>1) and not(word.isdigit()))]
    lemmatized_tokens=list(map(lemmatizer.lemmatize,tokens))
    string=""
    for word in lemmatized_tokens:
        string+=word+' '
    return string
    
df['text']=df[['title', 'company_profile', 'description','requirements','benefits']].fillna('').agg(' '.join, axis=1)
df['text_length']=df['text'].apply(lambda x:len(x.split()))
px.histogram(data_frame=df,x="text_length",color='fraudulent',marginal='rug')
df['text']=df['text'].apply(text_preprocess)
df['company_profile'].apply(lambda x:print(x) if 'see more' in str(x).lower() else False)
df[df['company_profile'].apply(lambda x:True if 'see more' in str(x).lower() else False)]['company_profile'].count()
x_train,x_test,y_train,y_test=train_test_split(df, df['fraudulent'], test_size=0.2,stratify=df['fraudulent'], random_state=42)
numerical_features=['telecommuting','has_company_logo','has_questions','location_count','null_count','text_length']

scaler=StandardScaler()
num_train=csr_matrix(scaler.fit_transform(x_train[numerical_features]))
num_test=csr_matrix(scaler.transform(x_test[numerical_features]))
ohe=OneHotEncoder()
ohe_train=ohe.fit_transform(x_train[['required_education','employment_type','required_experience']])
ohe_test=ohe.transform(x_test[['required_education','employment_type','required_experience']])
bow=CountVectorizer(ngram_range=(1,2))
text_train=bow.fit_transform(x_train['text'])
text_test=bow.transform(x_test['text'])
combined_train=hstack([num_train,ohe_train])
combined_test=hstack([num_test,ohe_test])
combined_train=hstack([combined_train,text_train])
combined_test=hstack([combined_test,text_test])
params={'C':[1,10,100,1000]}
clf=LogisticRegression(solver='liblinear',class_weight='balanced')
grid=GridSearchCV(clf,params,scoring='f1',n_jobs=-1,cv=5)
grid.fit(combined_train,y_train)
print(f1_score(y_train,grid.predict(combined_train)))
print(f1_score(y_test,grid.predict(combined_test)))
print(confusion_matrix(y_train,grid.predict(combined_train)))
print(confusion_matrix(y_test,grid.predict(combined_test)))
tfidf=TfidfVectorizer(ngram_range=(1,2))
text_train=tfidf.fit_transform(x_train['text'])
text_test=tfidf.transform(x_test['text'])
combined_train=hstack([num_train,ohe_train])
combined_test=hstack([num_test,ohe_test])
combined_train=hstack([combined_train,text_train])
combined_test=hstack([combined_test,text_test])
params={'C':[1,10,100,1000]}
clf=LogisticRegression(solver='liblinear',class_weight='balanced')
grid=GridSearchCV(clf,params,scoring='f1',n_jobs=-1,cv=5)
grid.fit(combined_train,y_train)
print(f1_score(y_train,grid.predict(combined_train)))
print(f1_score(y_test,grid.predict(combined_test)))
print(confusion_matrix(y_train,grid.predict(combined_train)))
print(confusion_matrix(y_test,grid.predict(combined_test)))
embeddings_index = {}
f = open('/kaggle/input/glove6b/glove.6B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Function to generate TF-IDF weighted glove embedding
def glove_embedding(x):
    embed=[]
    tfidf_dict=tfidf.vocabulary_
    x=list(x.split())
    for word in x:
        try:
            embed.append(embeddings_index[word]*tfidf_dict[word])
        except:
            continue
    return np.mean(embed,axis=0)
text_train=x_train['text'].apply(glove_embedding)
text_test=x_test['text'].apply(glove_embedding)
text_train=text_train.apply(pd.Series)
text_test=text_test.apply(pd.Series)
combined_train=hstack([num_train,ohe_train])
combined_test=hstack([num_test,ohe_test])
combined_train=hstack([combined_train,text_train])
combined_test=hstack([combined_test,text_test])
params={'C':[0.01,0.1,1,10]}
clf=LogisticRegression(solver='liblinear',class_weight='balanced')
grid=GridSearchCV(clf,params,scoring='f1',n_jobs=-1,cv=5)
grid.fit(combined_train,y_train)
print(f1_score(y_train,grid.predict(combined_train)))
print(f1_score(y_test,grid.predict(combined_test)))
print(confusion_matrix(y_train,grid.predict(combined_train)))
print(confusion_matrix(y_test,grid.predict(combined_test)))
clf=RandomForestClassifier(n_estimators=500,oob_score=True,n_jobs=-1,random_state=0)
clf.fit(text_train,y_train)
print(f1_score(y_train,clf.predict(text_train)))
print(f1_score(y_test,clf.predict(text_test)))
print(confusion_matrix(y_train,clf.predict(text_train)))
print(confusion_matrix(y_test,clf.predict(text_test)))
train_data = lgb.Dataset(combined_train, label=y_train)
test_data = lgb.Dataset(combined_test, label=y_test)
params = {}
params['learning_rate'] = 0.04
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['max_depth'] = 5
params['lambda_l1'] = 0
params['lambda_l2'] = 0
params['n_jobs'] = 4
params['class_weight']='balanced'

clf = lgb.train( params,train_data, 1000, valid_sets=[test_data])
thresh=0.20
print(f1_score(y_train,list(map(lambda x:1 if x>thresh else 0,clf.predict(combined_train)))))
print(f1_score(y_test,list(map(lambda x:1 if x>thresh else 0,clf.predict(combined_test)))))
print(confusion_matrix(y_train,list(map(lambda x:1 if x>thresh else 0,clf.predict(combined_train)))))
print(confusion_matrix(y_test,list(map(lambda x:1 if x>thresh else 0,clf.predict(combined_test)))))
# bert_df=df.iloc[:,[23,17]]
# bert_df.columns=['text','labels']
# bert_df['text']=bert_df['text'].apply(text_preprocess)
# train_df,eval_df=train_test_split(bert_df, test_size=0.2,stratify=bert_df['labels'], random_state=42)
#!pip install spacy==2.1.9
#!pip install simpletransformers wandb 
# from simpletransformers.classification import ClassificationModel

# args={"learning_rate": 3e-5,"save_steps": 4000,"manual_seed": 0,'save_model_every_epoch':False,'sliding_window': True,
#         'fp16': False, 
#         'train_batch_size': 64, 
#         'do_lower_case': True,
#         'overwrite_output_dir': True, 
#         'num_train_epochs':2
#      }

# model = ClassificationModel('roberta', 'roberta-base', num_labels=2,use_cuda=True,args=args)
# model.train_model(train_df)
# result, model_outputs, wrong_predictions = model.eval_model(train_df,acc=f1_score)
# print('[',result['tn'],',',result['fp'],']','\n[',result['fn'],',',result['tp'],']')
# print("f1_score: ",result['acc'])