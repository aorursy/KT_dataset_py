#Load libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import nltk
import os
import folium
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report, roc_curve, roc_auc_score, auc
#print(os.listdir("../input/donorschoose-application-screening"))
train = pd.read_csv("../input/train.csv", low_memory=False)
test = pd.read_csv("../input/test.csv", low_memory=False)
res = pd.read_csv("../input/resources.csv", low_memory=False)
### look at the shape
print("The train data has {} records with {} variables".format(*train.shape))
print("The test data has {} records with {} variables".format(*test.shape))
print("The resource dataset has {} records with {} variables".format(*res.shape))
## Read the first 3 lines
train.head(3)
### columns name
print(train.columns)
print(train.dtypes)
## create datatime with month and year
train['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
train['Month'] = train['project_submitted_datetime'].dt.month
train['Year'] = train['project_submitted_datetime'].dt.year
## missing values 
from IPython.display import HTML, display
import tabulate

df_Nan = train.isnull().sum().sort_values(ascending=False)/ len(train) * 100
df_Nan = pd.DataFrame(df_Nan)
display(HTML(tabulate.tabulate(df_Nan, tablefmt='html', headers=["Variable", "% of Nans"])))
train.describe()["teacher_number_of_previously_posted_projects"]
number_posted = pd.DataFrame({"teacher_number_of_previously_posted_projects":train["teacher_number_of_previously_posted_projects"], 
                              "log(x + 1)":np.log1p(train["teacher_number_of_previously_posted_projects"])})
number_posted.hist(figsize = (15,6));
plt.subplots(figsize=(10,6))
sns.countplot(x = 'project_is_approved', data = train, order = [1, 0])
plt.xlabel('Project approved')
plt.ylabel('Total number of projects')
plt.title('Project approved and not');
print("The project approved variable is unbalanced, in fact {} projects has been approved agaisnt \
{} not approved!".format(*train['project_is_approved'].value_counts()))
plt.subplots(figsize=(10,6))
sns.countplot(x ='teacher_prefix', data = train)
plt.xlabel('Teacher prefix')
plt.ylabel('Total number of Teacher');
plt.subplots(figsize=(15,8))
sns.countplot(x ='school_state', data = train, orient = 'h')
plt.xticks(rotation=45);
plt.xlabel('States')
plt.ylabel('Total number of States');
plt.subplots(figsize=(10,6))
sns.countplot(x ='project_grade_category', data = train)
plt.xlabel('Project grade category')
plt.ylabel('Total number of grade category');
plt.subplots(figsize=(10,6))
train['project_subject_categories'].value_counts().head(10).plot(kind = 'barh')
plt.xlabel('Total number of subject categories')
plt.ylabel('Subject categories');
### Approved and teacher
ap_teacher = pd.crosstab(train['teacher_prefix'], train['project_is_approved'], margins=True)
#ap_teacher
ap_teacher/ap_teacher.loc["All","All"]
plt.subplots(figsize=(10,6))
sns.countplot(y ='teacher_prefix', hue = 'project_is_approved', data = train, palette="Set3")
plt.ylabel('Teacher prefix');
ap_grade = pd.crosstab(train['project_grade_category'], train['project_is_approved'], margins=True)
#ap_grade
ap_grade/ap_grade.loc["All","All"]
plt.subplots(figsize=(10,6))
sns.countplot(y = 'project_grade_category', hue = 'project_is_approved', data = train, palette="Set2")
plt.ylabel('Project grade category');
sns.factorplot(x='teacher_prefix', hue='project_grade_category', col="project_is_approved",
                   data=train, kind="count", size=8, aspect=1);
train.set_index('project_submitted_datetime',inplace=True)
train.groupby("project_is_approved").resample('W')['teacher_number_of_previously_posted_projects'].sum().unstack('project_is_approved', fill_value=0).plot(figsize=(10, 6));
stopwords = set(STOPWORDS)
wd = WordCloud(width=512, height=512, stopwords=stopwords, background_color="skyblue", colormap="Reds"
              ).generate(' '.join(train.project_title))
plt.figure(figsize=(14,10))
plt.imshow(wd, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
## For project approved
text = ' '.join(list(train[train['project_is_approved'] == 1]['project_resource_summary']))
wordcloud = WordCloud(stopwords=stopwords, max_font_size=30, background_color="orange").generate(text)
plt.figure(figsize=(14,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
g = train[(train['teacher_prefix'] == 'Mrs.') & (train['project_is_approved'] == 0)]
g.groupby(['Month','project_grade_category']).size().unstack('project_grade_category',fill_value=0).plot(figsize=(10, 6));
g1 = train.groupby(['school_state', 'teacher_prefix']).size().unstack('teacher_prefix',fill_value=0)
g1.sort_values(['Ms.', 'Mrs.', 'Mr.', 'Teacher', 'Dr.'], ascending=False).head(20)[['Mrs.', 'Ms.', 'Mr.', 'Teacher', 'Dr.']].plot(kind='bar', figsize=(12, 7));