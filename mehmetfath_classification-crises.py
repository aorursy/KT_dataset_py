import matplotlib.pyplot as plt

import plotly.express as px

import pandas as pd

import seaborn as sns

import numpy as np

import pandas as pd

import warnings

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.utils import resample



warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

baslik_font = {'family': 'arial', 'color': 'darkred','weight': 'bold','size': 13 }

eksen_font  = {'family': 'arial', 'color': 'darkblue','weight': 'bold','size': 10 }
df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv', index_col='case')

df.head()
print("Checking the columns in the dataset.")

df.columns
df.info()
df.isnull().sum()
df.describe()
# Dataset is non-uniform and recently formed countries have least data

plt.figure(figsize=(8,8))

counts= df['country'].value_counts()

country=counts.index

explode = (0.2, 0.1, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0)

plt.pie(counts, explode=explode,labels=country,autopct='%1.1f%%')

plt.show()
# Let me visualize the country that has high number of banking_crisis 

plt.figure(figsize=(10,5))

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='banking_crisis')

plt.title('Graph of Country Representation based on Banking Crisis')

plt.xticks(rotation = 60)

plt.xlabel(None)

plt.show()
plt.figure(figsize=(20,14))

plt.subplot(221)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='systemic_crisis')

plt.title('Graph of Country Representation based on systemic_crisis')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.subplot(222)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='inflation_crises')

plt.title('Graph of Country Representation based on inflation_crises')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.subplot(223)

sns.set_style('whitegrid')

sns.countplot(x='country', data=df, palette='gist_rainbow', hue='banking_crisis')

plt.title('Graph of Country Representation based on Banking Crisis')

plt.xticks(rotation = 90)

plt.xlabel(None)

plt.tight_layout()

plt.show()

plt.figure(figsize=(15,8))

plt.subplot(121)

sns.barplot(x='systemic_crisis',y='country',data=df, palette='Paired')

plt.ylabel(None)

plt.title("Systemic_Crisis", fontdict=baslik_font)

plt.subplot(122)

plt.title("İnflation_Crises", fontdict=baslik_font)

sns.barplot(y='country',x='inflation_crises',data=df,palette='Paired')

plt.ylabel(None)

plt.tight_layout()

plt.show()
#The inflation and exchange rates are good indicator for economic health for the country

plt.figure(figsize=(15,8))

count = 1

for country in df.country.unique():

    plt.subplot(len(df.country.unique())/4,5,count)

    count+=1

    sns.lineplot(df[df.country==country]['year'],df[df.country==country]['exch_usd'], color="darkred")

    sns.lineplot(df[df.country==country]['year'],df[df.country==country]['inflation_annual_cpi'],color="darkblue")

    plt.subplots_adjust(wspace=0.4,hspace=0.5)

    plt.xlabel(None)

    plt.ylabel('İnflation/Exchange Rates')

    plt.title(country,baslik_font)
df["banking_crisis"]=df.banking_crisis.replace({'crisis':1,'no_crisis':0})

df.head(1)
df.corr()
df.columns
plt.figure(figsize=(10,7))

sns.heatmap(df.corr(), cmap='magma', annot=True)

plt.ylim(0,11)
a=df.sort_values(by=['year'])



fig = px.choropleth(a,locations="cc3",

                    color="exch_usd",animation_frame="year", 

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa exch_usd',

    geo_scope='africa', 

)
fig = px.choropleth(a,locations="cc3",

                    color="inflation_annual_cpi",animation_frame="year", 

                    hover_name="country", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa Enflasyon Durumu',

    geo_scope='africa', 

)
a1=df.groupby('country').sum()

a1['cc2']=['DZA', 'AGO', 'CAF', 'EGY','CIV',  'KEN', 'MUS', 'MAR', 'NGA',

       'ZAF', 'TUN', 'ZMB', 'ZWE']

a1['country1']=['Algeria', 'Angola', 'Central African Republic',

       'Egypt','Ivory Coast', 'Kenya', 'Mauritius', 'Morocco', 'Nigeria',

       'South Africa', 'Tunisia', 'Zambia', 'Zimbabwe']


fig = px.choropleth(a1,locations="cc2",

                    color="banking_crisis",

                    hover_name="country1", 

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(

    title_text = 'Africa Enflasyon Durumu',

    geo_scope='africa',

)
log_reg = LogisticRegression()
df.corr().banking_crisis.abs().sort_values()
X = df[['year', 'systemic_crisis', 'domestic_debt_in_default', 'sovereign_external_debt_default',

       'gdp_weighted_default',  'independence',

       'currency_crises','exch_usd', 'inflation_crises','inflation_annual_cpi']]

y = df['banking_crisis']
X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=111)
log_reg.fit(X_eğitim, y_eğitim)
egitim_dogruluk = log_reg.score(X_eğitim, y_eğitim)

test_dogruluk = log_reg.score(X_test, y_test)

print('One-vs-rest', '-'*20, 

      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk), 

      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')
log_reg_mnm = LogisticRegression(multi_class='multinomial', solver='lbfgs')

log_reg_mnm.fit(X_eğitim, y_eğitim)

egitim_dogruluk = log_reg_mnm.score(X_eğitim, y_eğitim)

test_dogruluk = log_reg_mnm.score(X_test, y_test)

print('Multinomial (Softmax)', '-'*20, 

      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk), 

      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')
tahmin_değerleri = np.array([[2025,1,1,0,1,0,1.5,1,1,0]])

print(log_reg.predict(tahmin_değerleri))
C_değerleri = [0.001,0.01,0.1,1,10,100, 1000]

dogruluk_df = pd.DataFrame(columns = ['C_Değeri','Doğruluk'])



dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])



for c in C_değerleri:

    

    # Apply logistic regression model to training data

    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0)

    lr.fit(X_eğitim,y_eğitim)

    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,

                                                    'Eğitim Doğruluğu' : lr.score(X_eğitim, y_eğitim),

                                                    'Test Doğruluğu': lr.score(X_test, y_test)

                                                    }, ignore_index=True)

display(dogruluk_değerleri)    
X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=111)

bankingcrisis_modeli = LogisticRegression()

bankingcrisis_modeli.fit(X_eğitim, y_eğitim)



tahmin_eğitim =bankingcrisis_modeli.predict(X_eğitim)

tahmin_test = bankingcrisis_modeli.predict(X_test)
tahmin_test_ihtimal = bankingcrisis_modeli.predict_proba(X_test)[:,1]
from sklearn.metrics import confusion_matrix

hata_matrisi_eğitim = confusion_matrix(y_eğitim, tahmin_eğitim)

hata_matrisi_test = confusion_matrix(y_test, tahmin_test)

print("Hata Matrisi (Eğitim verileri)", "-"*30, hata_matrisi_eğitim, sep="\n")

print("Hata Matrisi (Test verileri)", "-"*30, hata_matrisi_test, sep="\n")
TN = hata_matrisi_test[0][0]

TP = hata_matrisi_test[1][1]

FP = hata_matrisi_test[0][1]

FN = hata_matrisi_test[1][0]



print("Doğru negatif sayısı   :", TN)

print("Doğru pozitif sayısı   :", TP)

print("Yanlış pozitif sayısı  :", FP)

print("Yanlış negatif sayısı  :", FN)
from sklearn.metrics import accuracy_score



print("Modelden alınan doğruluk değeri : ",  bankingcrisis_modeli.score(X_test, y_test))

print("Hesaplanan doğruluk değeri      : ",  (TN + TP)/(FN + FP + TN + TP))

print("accuracy_score() değeri         : ",  accuracy_score(y_test, tahmin_test))
from sklearn.metrics import precision_score



print("Hesaplanan doğruluk değeri      : ",  (TP)/(FP + TP))

print("precision_score() değeri        : ",  precision_score(y_test, tahmin_test))
from sklearn.metrics import recall_score



print("Hesaplanan doğruluk değeri   : ",  (TP)/(TP + FN))

print("recall_score() değeri        : ",  recall_score(y_test, tahmin_test))
print("Hesaplanan özgünlük değeri   : ",  (TN)/(TN + FP))
from sklearn.metrics import f1_score



hassasiyet_degeri = precision_score(y_test, tahmin_test)

duyarlılık_değeri = recall_score(y_test, tahmin_test)





print("Hesaplanan f1 skoru   : ",  2*((hassasiyet_degeri*duyarlılık_değeri)/(hassasiyet_degeri + duyarlılık_değeri)))

print("f1_score() değeri     : ",  f1_score(y_test, tahmin_test))
from sklearn.metrics import classification_report, precision_recall_fscore_support



print(classification_report(y_test,tahmin_test) )



print("f1_score() değeri        : {:.2f}".format(f1_score(y_test, tahmin_test)))

print("recall_score() değeri    : {:.2f}".format(recall_score(y_test, tahmin_test)))

print("precision_score() değeri : {:.2f}".format(precision_score(y_test, tahmin_test)))

print('\n')



metrikler =  precision_recall_fscore_support(y_test, tahmin_test)

print("Hassasiyet :" , metrikler[0]) 

print("Duyarlılık :" , metrikler[1]) 

print("F1 Skoru   :" , metrikler[2]) 
from sklearn.metrics import roc_curve, roc_auc_score



fpr, tpr, thresholds  = roc_curve(y_test, tahmin_test_ihtimal)



import matplotlib.pyplot as plt

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
print('AUC Değeri : ', roc_auc_score(y_test, tahmin_test_ihtimal))
from sklearn.metrics import precision_recall_curve

hassasiyet, duyarlılık, _ = precision_recall_curve(y_test, tahmin_test_ihtimal)



plt.plot(duyarlılık, hassasiyet)

plt.show()
C_değerleri = [0.001,0.01,0.1,1,10,100, 1000]

dogruluk_df = pd.DataFrame(columns = ['C_Değeri','Doğruluk'])



dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])



for c in C_değerleri:

    

    # Apply logistic regression model to training data

    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0)

    lr.fit(X_eğitim,y_eğitim)

    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,

                                                    'Eğitim Doğruluğu' : lr.score(X_eğitim, y_eğitim),

                                                    'Test Doğruluğu': lr.score(X_test, y_test)

                                                    }, ignore_index=True)

display(dogruluk_değerleri)  
for c in C_değerleri:



    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0)

    lr.fit(X_eğitim,y_eğitim)

    tahmin_test_ihtimal = lr.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds  = roc_curve(y_test, tahmin_test_ihtimal)

    i=0

    

    plt.figure(figsize=(5,3))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(c)

    

plt.show()
def model_olustur(X, y):

    X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X, y, test_size=0.20, random_state=111, stratify = y)

    logreg_model = LogisticRegression()

    logreg_model.fit(X_eğitim, y_eğitim)



    tahmin_eğitim = logreg_model.predict(X_eğitim)

    tahmin_test = logreg_model.predict(X_test)

    hata_matrisi_eğitim = confusion_matrix(y_eğitim, tahmin_eğitim)

    hata_matrisi_test = confusion_matrix(y_test, tahmin_test)

    print("Modelin doğruluk değeri : ",  logreg_model.score(X_test, y_test))

    print("Eğitim veri kümesi")

    print(classification_report(y_eğitim,tahmin_eğitim) )

    print("Test veri kümesi")

    print(classification_report(y_test,tahmin_test) )

    return  None
model_olustur(X,y)
from sklearn.utils import resample
normal_alısveris = df[df.banking_crisis == 0]

sahte_alısveris = df[df.banking_crisis == 1]



sahte_alısveris_artırılmış = resample(sahte_alısveris,

                                     replace = True,

                                     n_samples = len(normal_alısveris),

                                     random_state = 111)



artırılmıs_df = pd.concat([normal_alısveris, sahte_alısveris_artırılmış])

artırılmıs_df.banking_crisis.value_counts()
X = artırılmıs_df[['year', 'systemic_crisis', 'domestic_debt_in_default', 'sovereign_external_debt_default',

       'gdp_weighted_default',  'independence',

       'currency_crises','exch_usd', 'inflation_crises','inflation_annual_cpi']]

y = artırılmıs_df['banking_crisis']

model_olustur(X,y)
