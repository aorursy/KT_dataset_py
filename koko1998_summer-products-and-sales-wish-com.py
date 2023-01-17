# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_products=pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
df_products.head()
df_products.columns
df_products.info()
product_color=df_products['product_color'].value_counts()
print(product_color.to_string())
#for index,item in df_products['product_color'].items() :
   # item=str(item).lower()
df_products['product_color']=df_products['product_color'].str.lower()
print(df_products['product_color'].value_counts().to_string())
df_products['product_color'].replace('grey','gray',inplace=True)
df_products['product_color']=df_products['product_color'].str.split('&')
print(df_products['product_color'])
df_products['product_color']=df_products['product_color'].str[0]
print(df_products['product_color'].value_counts().to_string())
df_products['product_color']=df_products['product_color'].str.split(' ')
df_products['product_color']=df_products['product_color'].str[0]
print(df_products['product_color'].value_counts().to_string())
df_products['product_color']=df_products['product_color'].str.split('-')
df_products['product_color']=df_products['product_color'].str[0]
print(df_products['product_color'].value_counts().to_string())
df_products['product_color'].replace({'applegreen':'green','blackwhite':'black',
                                     'lightgreen':'green','lightblue':'blue','navyblue':'blue',
                                     'denimblue':'blue','prussianblue':'blue','lakeblue':'blue',
                                     'darkblue':'blue','skyblue':'blue','lightblue':'blue'
                                     ,'navyblue':'blue','armygreen':'green','darkgreen':'green',
                                     'applegreen':'green','lightgreen':'green','mintgreen':'green',
                                      'winered':'red','coralred':'red','lightred':'red',
                                     'watermelonred':'red','fluorescentgreen':'green',
                                     'rosered':'rose','rosegold':'rose','greysnakeskinprint':'gray',
                                     'lightgray':'gray','lightgrey':'gray','whitefloral':'white'
                                     ,'whitestripe':'white','lightpink':'pink','dustypink':'pink',
                                     'lightkhaki':'khaki','lightpurple':'purple','lightyellow':'yellow',
                                     'tan':'brown','coffee':'brown','offwhite':'white',
                                     'offblack':'black','coolblack':'black','leopardprint':
                                     'leopard','claret':'red','violet':'blue','apricot':'pink',
                                     'nude':'pink','gold':'brown','ivory':'beige','burgundy':'red',
                                     'navy':'blue','wine':'red','camel':'brown','khaki':'brown','light':'white'},inplace=True)
df_products['product_color'].replace({'floral':'multicolor','leopard':'multicolor',
                                     'army':'multicolor','camouflage':'multicolor',
                                     'silver':'multicolor','star':'multicolor',
                                     'khaki':'multicolor','rainbow':'multicolor',
                                     'jasper':'multicolor',} ,inplace=True)
df_products['currency_buyer'].value_counts()
print(df_products['product_color'].value_counts().to_string())
df_products.drop('currency_buyer',axis=1,inplace=True)
df_products['merchant_id'].value_counts()
df_products[df_products.duplicated()]
df_products.drop('merchant_title',axis=1,inplace=True)
df_products.drop('merchant_name',axis=1,inplace=True)
df_products['origin_country'].value_counts()
df_products.drop('merchant_profile_picture',axis=1,inplace=True)
df_products['crawl_month'].value_counts()
df_products.drop('crawl_month',axis=1,inplace=True)
df_products['urgency_text'].value_counts()
df_products['urgency_text'].isnull().sum()
df_products.drop('urgency_text',axis=1,inplace=True)
df_products.drop('merchant_info_subtitle',axis=1,inplace=True)
df_products.drop('product_picture',axis=1,inplace=True)
df_products.drop('product_url',axis=1,inplace=True)
df_products['theme'].value_counts()
df_products.drop('theme',axis=1,inplace=True)
df_products.head()
df_products.columns
df_products.info()
df_products.drop('has_urgency_banner',axis=1,inplace=True)
df_products['origin_country'][df_products['origin_country']=="NaN"]
df_products['origin_country'].value_counts()
mode=df_products['origin_country'].mode()
print(mode)
df_products['origin_country']=df_products['origin_country'].replace(np.nan,"CN")
df_products['origin_country'].isnull().sum()
df_products.info()
mode=df_products['product_color'].mode()
print(mode)
df_products['product_color']=df_products['product_color'].replace(np.nan,"black")
df_products['origin_country'].isnull().sum()
df_products['product_id'].value_counts()
averagefeatur=df_products.groupby(df_products['product_id']).mean()[['price','retail_price','merchant_rating',
                                                                 'rating']]
averagefeatur
df_products.to_csv("df_products.csv")
df_products['diffrencebetprices']=df_products['retail_price']-df_products['price']
df_products['diffrencebetprices'].loc[df_products['diffrencebetprices']<=0]
print(df_products['product_variation_size_id'].value_counts().to_string())
df_products['product_variation_size_id']=df_products['product_variation_size_id'].str.lower()
print(df_products['product_variation_size_id'].value_counts().to_string())
df_products['product_variation_size_id'].replace({"s.":"s","size s":"s","size-s":"s","size s.":"s",
                                                "s(bust 88cm)":"s","suit-s":"s",
                                                "size--s":"s","size/s":"s","25-s":"s",
                                                "s(pink & black)":"s","us-s":"s",
                                                "pants-s":"s","s diameter 30cm":"s","s..":"s","s (waist58-62cm)":"s"
                                                , "s pink":"s","m.":"m","size m":"m","sizel":"l","32/l":"l"
                                                 ,"size-l":"l","x   l":"xl","xs.":"xs","size-xs":"xs","size xs":"xs"
                                                 ,"size-xxs":"xxs","size -xxs":"xxs","size xxs":"xxs"
                                                 ,"2xl":"xxl","xxxxxl":"5xl","size-5xl":"5xl","xxxxl":"4xl",
                                                 "xxxl":"3xl","size-4xl":"4xl","l.":"l",
                                                 "1 pc - xl":"xl","04-3xl":"3xl","size4xl":"4xl",
                                                  "us 6.5 (eu 37)":"m","us5.5-eu35":"s","26(waist 72cm 28inch)":
                                                  "3x","women size 36":"s","36":"s","women size 37":"m","eu 35":"s",
                                                  "3x":"3xl","eu39(us8)":"m","choose a size":"s",
                                                  "base & top & matte top coat":"s","4-5 years":"s",
                                                  "daughter 24m":"s","2":"s","5":"s",
                                                  "30 cm":"s","white":"s","round":"s","40 cm":"s","17":"l",
                                                  "60":"l","floating chair for kid":"s","au plug low quality":"s",
                                                  "baby float boat":"s","4":"xs","s/m(child)":"s","1pc":"s"
                                                  ,"3 layered anklet":"s","base coat":"s","20pcs-10pairs":"s",
                                                  "pack of 1":"s","100 x 100cm(39.3 x 39.3inch)":"s","b":"s",
                                                  "first  generation":"s","3x":"3xl","100 cm":"s","2pcs":"s",
                                                  "5pairs":"s","10pcs":"s","20pcs":"s","100pcs":"s",
                                                  "1m by 3m":"s","h01":"s","80 x 200 cm":"s","1":"xxxs",
                                                  "33":"xs","34":"xs","29":"4xl","25":"xxl","35":"s","1 pc.":"s",
                                                  "one size":"s","10 ml":"s"
                                                 },inplace=True)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x=df_products['product_variation_size_id'].value_counts().index
y=df_products['product_variation_size_id'].value_counts()
ax.bar(x, y, align='center')
ax.set_xticks(x)
ax.set_xlabel('Size')
ax.set_title('size counts')
plt.show()
df_products.to_csv("df_products.csv")
df_products['shipping_option_name'].value_counts()
standardshippinglist=["Livraison standard","Envio Padrão","Expediere Standard","Envío normal",
                     "الشحن القياسي","Standardversand","Standardowa wysyłka","Стандартная доставка",
                     "Standart Gönderi","Spedizione standard","การส่งสินค้ามาตรฐาน","ការដឹកជញ្ជូនតាមស្តង់ដារ"]
df_products.loc[df_products['shipping_option_name'] == "Livraison standard", "buymentlang"] = "French"
df_products.loc[df_products['shipping_option_name'] == "الشحن القياسي", "buymentlang"] = "Arabic"
df_products.loc[df_products['shipping_option_name'] == "Standard Shipping", "buymentlang"] = "English"
df_products.loc[df_products['shipping_option_name'] == "Envio Padrão", "buymentlang"] = "Portuguese"
df_products.loc[df_products['shipping_option_name'] == "Expediere Standard", "buymentlang"] = "Romanian"
df_products.loc[df_products['shipping_option_name'] == "Envío normal", "buymentlang"] = "Spanish"
df_products.loc[df_products['shipping_option_name'] == "Standardversand", "buymentlang"] = "German"
df_products.loc[df_products['shipping_option_name'] == "Livraison Express", "buymentlang"] = "French"
df_products.loc[df_products['shipping_option_name'] == "Standardowa wysyłka", "buymentlang"] = "Polish"
df_products.loc[df_products['shipping_option_name'] == "Стандартная доставка", "buymentlang"] = "Russian"
df_products.loc[df_products['shipping_option_name'] == "Standart Gönderi", "buymentlang"] = "Turkish"
df_products.loc[df_products['shipping_option_name'] == "Spedizione standard", "buymentlang"] = "Italian"
df_products.loc[df_products['shipping_option_name'] == "การส่งสินค้ามาตรฐาน", "buymentlang"] = "Thailand"
df_products.loc[df_products['shipping_option_name'] == "ការដឹកជញ្ជូនតាមស្តង់ដារ", "buymentlang"] = "Khmer"
df_products.loc[df_products['shipping_option_name'] == "Ekspresowa wysyłka", "buymentlang"] = "Polish"
df_products['buymentlang'].value_counts()
df_products['buymentlang'].isnull().sum()
#standardshippinglist
df_products['shipping_option_name'].replace(standardshippinglist,"Standard Shipping",inplace=True)
df_products['shipping_option_name'].value_counts()
Expresslist=["Livraison Express","Ekspresowa wysyłka"]
df_products['shipping_option_name'].replace(Expresslist,"Express Shipping",inplace=True)
df_products['shipping_option_price'].describe()
fig, ax = plt.subplots()
x=df_products['shipping_option_price'].value_counts().index
y=df_products['shipping_option_price'].value_counts()
ax.bar(x, y, align='center')
ax.set_xticks(x)
ax.set_xlabel('price shipping')
ax.set_title('price shipping counts')
plt.show()
df_products.loc[df_products['shipping_option_price']==12]['shipping_option_name']
df_products.loc[df_products['shipping_option_price']==7]['shipping_option_name']
df_products['countries_shipped_to'].value_counts()[:20]
import seaborn as sns 
sns.countplot(df_products['countries_shipped_to'].value_counts()[:20])
df_products["badge_local_product"].value_counts()
df_products["badge_product_quality"].value_counts()
df_products["badge_fast_shipping"].value_counts()
df_products["inventory_total"].value_counts()
df_products.to_csv("df_products.csv")
df_products=pd.get_dummies(df_products,columns=['product_color','shipping_option_name','origin_country','buymentlang'])
df_products.to_csv("df_products.csv")
print(x)
from wordcloud import WordCloud
text=list(df_products['title'])
textcol=''
for a in text :
    textcol=textcol+a
wordcloudmodel=WordCloud().generate(textcol)
plt.imshow(wordcloudmodel,interpolation='bilinear')
#plt.axis("on")
plt.show()
text=list(df_products['title'])
textcol=''
for a in text :
    textcol=textcol+a
wordcloudmodel=WordCloud().generate(textcol)
plt.imshow(wordcloudmodel,interpolation='bilinear')
plt.show()
df_products['product_variation_size_id'].value_counts()
#from sklearn import preprocessing 
#le =preprocessing.LabelEncoder()
#le.fit_transform(df_products['product_variation_size_id'].astype(str))
df_products=pd.get_dummies(df_products,columns=['product_variation_size_id'])
df_products['rating']=df_products['rating'].apply(round)
sns.barplot(df_products['rating'].value_counts().index,df_products['rating'].value_counts())
df_products['merchant_rating']=df_products['merchant_rating'].apply(round)
sns.barplot(df_products['merchant_rating'].value_counts().index ,df_products['merchant_rating'].value_counts())
df_products['product_variation_inventory'].value_counts()
df_products['inventory_total'].value_counts()
df_products['units_sold'].value_counts()
dropval=df_products.loc[df_products['units_sold']>60000]
dropval.shape
df_products=df_products.loc[df_products['units_sold']<=60000]
#df_products.drop(df_products[df_products['units_sold']>60000].index,axis=0,inplace=True)#
#print(,)
df_products.columns
X=[ 'price','uses_ad_boosts', 'rating', 'badges_count', 'badge_local_product','badge_product_quality',
   'badge_fast_shipping', 'shipping_option_price',
       'shipping_is_express', 'countries_shipped_to','merchant_rating',
       'merchant_has_profile_picture',
       'product_color_beige', 'product_color_black', 'product_color_blue',
       'product_color_brown', 'product_color_gray', 'product_color_green',
       'product_color_multicolor', 'product_color_orange',
       'product_color_pink', 'product_color_purple', 'product_color_red',
       'product_color_rose', 'product_color_white', 'product_color_yellow',
       'shipping_option_name_Express Shipping',
       'shipping_option_name_Standard Shipping', 'origin_country_AT',
       'origin_country_CN', 'origin_country_GB', 'origin_country_SG',
       'origin_country_US', 'origin_country_VE', 'buymentlang_Arabic',
       'buymentlang_English', 'buymentlang_French', 'buymentlang_German',
       'buymentlang_Italian', 'buymentlang_Khmer', 'buymentlang_Polish',
       'buymentlang_Portuguese', 'buymentlang_Romanian', 'buymentlang_Russian',
       'buymentlang_Spanish', 'buymentlang_Thailand', 'buymentlang_Turkish',
       'product_variation_size_id_3x', 'product_variation_size_id_3xl',
       'product_variation_size_id_4xl', 'product_variation_size_id_5xl',
       'product_variation_size_id_6xl', 'product_variation_size_id_l',
       'product_variation_size_id_m', 'product_variation_size_id_s',
       'product_variation_size_id_xl', 'product_variation_size_id_xs',
       'product_variation_size_id_xxl', 'product_variation_size_id_xxs',
       'product_variation_size_id_xxxs'
       ]
X=df_products[X]
Y=df_products['units_sold']
#X['product_variation_size_id']
X.shape
Y.shape
X.to_csv("X.csv")
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
scaler = StandardScaler()
lin = LinearRegression()
def modelfit(X,Y,scaler,modelalgorithm,train_test_split):
    #scaler.fit_transform(X,Y)
    normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
    scaler.fit_transform(X,Y)
    #scaler.fit_transform()
    X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=.2,random_state=42)
    modelalgorithm.fit(X_train,y_train)
    prediction =modelalgorithm.predict(X_test)
    #print(mean_absolute_error(y_test,prediction))
    return r2_score(y_test,prediction)
modelfit(X,Y,scaler,lin,train_test_split)

#accuracy=[]
#for i in range(X.shape[1]):
#    X_model=X.iloc[:,i:-1]
 ##   print("this model use all data without first",i,"columns")
  #  ac=modelfit(X_model,Y,scaler,lin,train_test_split)
  #  accuracy.append(ac)
accuracy=[]
Start=[]
End=[]
maxac=modelfit(X.iloc[:,0:1],Y,scaler,lin,train_test_split)
index_i=0
index_j=0
for i in range(1,X.shape[1]):
    for j in range(1,X.shape[1]):
        if (i>j):
            X_model=X.iloc[:,j:i]
            print("this model use data start from",j,"column to",i ,"column")
            try :
                ac=modelfit(X_model,Y,scaler,lin,train_test_split)
                accuracy.append(ac)
                Start.append(j)
                End.append(i)
                if ac>=maxac:
                    maxac=ac
                    index_i=i
                    index_j=j
                print(ac)
            except ValueError :
                print("error")
            
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Start,End,accuracy)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
index_i
index_j
max(accuracy)
min(accuracy)
X_predict=X.iloc[:,49:59]
y_result=Y.iloc[:]
#print(X.iloc[25,7:12])
scaler.fit_transform(X_predict,Y)
X_train, X_test, y_train, y_test=train_test_split(X_predict,Y,test_size=.2,random_state=42)
lin.fit(X_train,y_train)
prediction =lin.predict(X.iloc[:,49:59])
#print(mean_absolute_error(y_test,prediction))
print(prediction,y_result)
plt.plot(prediction)
plt.show()
plt.plot(y_result)
plt.show()
prediction.mean()
y_result.mean()
col=[]
accuracy=[]
for j in range(1,X.shape[1]):
    X_model=X.iloc[:,j:j+1]
    print("this model use data start from",j,"column to")
    ac=modelfit(X_model,Y,scaler,lin,train_test_split)
    print(ac)
    accuracy.append(ac)
    col.append(j)
plt.plot(col,accuracy)
plt.show()