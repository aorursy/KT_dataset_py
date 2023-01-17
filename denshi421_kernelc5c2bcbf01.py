%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score,recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D #3D散布図の描画
import itertools #組み合わせを求めるときに使う
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
df_csv = pd.read_csv("../input/survey.csv")
print(df_csv.columns)
display(df_csv.head())
display(df_csv.tail())
# データの閲覧
# 明らかに関係のないTimestampは除く。
df_data = df_csv.drop(['Timestamp'], axis=1)

for col in df_data.columns:
    if col=="treatment":
        continue
    df_data[col].replace('All','all', inplace=True)
    df_gender = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True)
    display(df_gender.T.sort_values("All", ascending=False))
#欠測値の数
pd.DataFrame(df_data.isnull().sum(), columns=['num of missing'])
df_data = df_data.drop(['state','comments'], axis=1)
df_data.columns
df_data['self_employed'] = df_data['self_employed'].fillna('Non-Response')
df_data['work_interfere'] = df_data['work_interfere'].fillna('Non-Response')
df_data
col = "Age"
df_age = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True)
df_age
#0歳以下72歳以上は異常値、10歳未満は先天性の精神病である可能性を考え、10歳以上を対象とした。
#異常値の置き換え
df_data['Age'] = df_data['Age'].map(lambda x: x if 10<x<80 else 0)
display(df_data.sort_values(by='Age').head())
display(df_data.sort_values(by='Age').tail())
col = "Age"
df_age = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True)
df_age
col = 'Gender'
df_data[col].replace('All','all', inplace=True)
df_gender = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True)
display(df_gender.T.sort_values("All", ascending=False))
df_data['Gender'] = df_data['Gender'].map(lambda x: str(x).strip()) #文字列の前後の空白削除
df_data['Gender2'] = df_data['Gender'].map({ \
                                            'Male':'Male', \
                                            'male':'Male', \
                                            'M':'Male', \
                                            'm':'Male', \
                                            'Make':'Male', \
                                            'Male':'Male', \
                                            'Cis Male':'Male', \
                                            'cis male':'Male', \
                                            'Cis Man':'Male', \
                                            'Male (CIS)':'Male', \
                                            'Man':'Male', \
                                            'Mal':'Male', \
                                            'maile':'Male', \
                                            'ostensibly male, unsure what that really means':'Male', \
                                            'Mail':'Male', \
                                            'msle':'Male', \
                                            'something kinda male?':'Male', \
                                            'male leaning androgynous':'Male', \
                                            'Guy (-ish) ^_^':'Male', \
                                            'Trans-female':'Male', \
                                            'Trans woman':'Male', \
                                            'Female (trans)':'Male', \
                                            'Male-ish':'Male', \
                                            'Malr':'Male', \
                                            \
                                            'Female':'Female', \
                                            'female':'Female', \
                                            'F':'Female', \
                                            'f':'Female', \
                                            'Female':'Female', \
                                            'Woman':'Female', \
                                            'woman':'Female', \
                                            'Female(trans)':'Female', \
                                            'cis-female/femme':'Female', \
                                            'Cis Female':'Female', \
                                            'femail':'Female', \
                                            'Female (cis)':'Female', \
                                            'Femake':'Female', \
                                            'Female (cis)':'Female', \
                                            \
                                            #性別がわからない奴は'Agender'に置換
                                            'non-binary':'Agender', \
                                            'p':'Agender', \
                                            'fluid':'Agender', \
                                            'queer':'Agender', \
                                            'queer/she/they':'Agender', \
                                            'Genderqueer':'Agender', \
                                            'A little about you':'Agender', \
                                            'all':'Agender', \
                                            'Androgyne':'Agender', \
                                            'Agender':'Agender', \
                                            'Enby':'Agender', \
                                            'Neuter':'Agender', \
                                            'Nah':'Agender' \
                                           })


col = 'Gender2'
df_gender2 = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True)
print(df_gender['All'])
print(df_gender2['All'])

df_gender2 = df_gender2.drop(['All'], axis=1)
df_gender2 = df_gender2.drop(['All'])
display(df_gender2)
df_gender2.plot.bar(stacked=True)
check = df_data[['Gender', 'Gender2']]
display(check.sort_values(by='Gender2'))
df_data = df_data.drop(['Gender'], axis=1)
df_data.columns
#並び替え
df_data = df_data[['treatment','Age', 'Gender2', 'Country', 'self_employed', 'family_history', 
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']]
df_data.columns
df_r = pd.get_dummies(df_data) #one-hotベクトルに変換
display(df_r.columns)
display(df_r.head())
display(df_r.corr().sort_values(by='treatment_Yes').head(10))
display(df_r.corr().sort_values(by='treatment_Yes').tail(10))
display(df_r.corr().sort_values(by='treatment_No').head(10))
display(df_r.corr().sort_values(by='treatment_No').tail(10))
def Age(x):
    if 10 <= x < 20:
        x = 10
    elif 20 <= x < 30:
        x = 20
    elif 30 <= x < 40:
        x = 30
    elif 40 <= x < 50:
        x = 40
    elif 50 <= x < 60:
        x = 50
    elif 60 <= x < 70:
        x = 60
    elif 70 <= x < 80:
        x = 70
    return x
        
df_data['Age2'] = df_data['Age'].map(lambda x: Age(x))
df_data = df_data[['treatment','Age', 'Age2','Gender2', 'Country', 'self_employed', 'family_history', 
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']]
for col in df_data.columns:
    if col=='treatment':
        continue
    df_Graph = pd.crosstab(index=df_data["treatment"], columns=df_data[col], margins=True, normalize=True)
    df_Graph2 = df_Graph.drop(['All'], axis=1)
    df_Graph2 = df_Graph2.drop(['All'])
    display(df_Graph.T.sort_values("All", ascending=False))
    df_Graph2.plot.barh(stacked=True)
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=3, title=col)
    plt.show()
