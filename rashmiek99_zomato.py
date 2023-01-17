import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator



from PIL import Image



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings('ignore')
zomato_data = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')

zomato_data.head().T
print('Number of rows    =',zomato_data.shape[0])

print('Number of columns =',zomato_data.shape[1])

print("Checking the info:\n")

print(zomato_data.info())
print("How many null values are there?\n",zomato_data.isnull().sum())
print("Dropping all the null value data")

zomato_data.dropna(how='any',inplace=True)
print("Checking unique values in the data\n")

print(zomato_data.nunique())
print("Changing columns to more meaningful name")

zomato_data.columns = ['url', 'address', 'name', 'online_order','book_table', 'rate', 'votes',

       'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',

       'avg_cost', 'reviews_list', 'menu_item',

       'type', 'locality']
print("Some columns are uneccessary for us like url,location,phone,menu_item")

zomato_data.drop(['url','location','phone',

                          'menu_item'],inplace=True,axis=1)
#changing number  line 1,200 to 1200 for computing

zomato_data['avg_cost'] = zomato_data['avg_cost'].str.replace(',', '').astype(float)#



#Removing whitespaces

zomato_data['name'] = zomato_data['name'].str.strip()

zomato_data['locality'] = zomato_data['locality'].str.strip()

zomato_data['type'] = zomato_data['type'].str.strip()

zomato_data['cuisines'] = zomato_data['cuisines'].str.strip()
bg_color = (0.25, 0.25, 0.25)

sns.set(rc={"font.style":"normal",

            "axes.facecolor":bg_color,

            "figure.facecolor":bg_color,

            "text.color":"white",

            "xtick.color":"white",

            "ytick.color":"white",

            "axes.labelcolor":"white",

            "axes.grid":False,

            'axes.labelsize':25,

            'figure.figsize':(15.0,15.0),

            'xtick.labelsize':15,

            'ytick.labelsize':15})    
cost_for_two = pd.cut(zomato_data['avg_cost'],bins = [0, 200, 500, 1000, 5000, 8000],labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000',])

ax = sns.countplot(cost_for_two, palette = sns.color_palette('magma', 5))

plt.title("Average Cost for 2 people",fontsize=15,fontweight='bold')

plt.yticks([])

plt.ylabel("")

plt.xlabel('Cost',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.25,p.get_height()))


sns.countplot(zomato_data['locality'].sort_values(),palette='muted')

plt.title('Number of Resturants based on locality',fontsize=15,fontweight='bold')

plt.xticks(rotation=90,fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold')

plt.xlabel('Locality',fontsize=10,fontweight='bold')

plt.ylabel('Number of Resturants',fontsize=10,fontweight='bold');

sns.countplot(zomato_data['type'].sort_values(),palette='cubehelix')

plt.title('Restuarant Type',fontsize=15,fontweight='bold')

plt.xticks(rotation=45,ha='right',fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold')

plt.ylabel('Number of Resturants',fontsize=10,fontweight='bold')

plt.xlabel('Restuarant Type',fontsize=10,fontweight='bold');
ax = sns.countplot(x=zomato_data['online_order'],hue=zomato_data['online_order'],palette='Set1')

plt.title('Online Orders')



plt.ylabel('')

plt.yticks([])



for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x() + 0.15,p.get_height()),fontsize=10,fontweight='bold')



ax.set_xticklabels(['Yes                                 ','                          No']);
X = zomato_data

X['rate'] = X['rate'].astype(str)

X['rate'] = X['rate'].str.strip()

X['rate'] = X['rate'].apply(lambda x: x.replace('/5',''))

X['rate'] = X['rate'].apply(lambda x: x.replace('NEW','0'))

X['rate'] = X['rate'].astype(float)
x = pd.crosstab(zomato_data['rate'],zomato_data['book_table'])

x.plot(kind='bar',stacked=True)



plt.title('Table Booking - Rating',fontsize=15,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold')

plt.ylabel('Number of Table Bookings',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold');
x = pd.crosstab(zomato_data['rate'],zomato_data['online_order'])

x.plot(kind='bar',stacked=True);

plt.title('Online Order - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Number of Online Orders',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');
x = pd.crosstab(zomato_data['rate'],cost_for_two)

x.plot(kind='bar',stacked=True);

plt.title('Avg cost - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Average Cost',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');
my_mask = np.array(Image.open("../input/mymask/f.jpg"))

my_words=''



my_dishes = zomato_data[zomato_data['rate']>4]

for dishes in my_dishes['dish_liked'].unique():

    val = str(dishes)

    tokens = val.split()

    

    for words in tokens:

        my_words = my_words + words +' '

    

wc = WordCloud(background_color="white", max_words=100, mask=my_mask,

               contour_width=3, contour_color='steelblue')



wc.generate(my_words)

plt.figure()

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show()
stop_words=set(STOPWORDS)

my_words=''



my_cuisines = zomato_data[zomato_data['rate']>4]

for cuisines in my_cuisines['cuisines'].unique():

    val = str(cuisines)

    tokens = val.split()

    

    for words in tokens:

        my_words = my_words + words +' '



plt.title("Popular Cuisines",fontsize=35)



wordcloud = WordCloud(width=800,height=800,max_words=50,collocations=False,background_color='white',

        stopwords= stop_words,contour_width=3, colormap='magma',min_font_size=10).generate(my_words)



plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
my_list = ['North Indian', 'Mughlai', 'Chinese','Thai','Mexican', 'Italian','South Indian','Continental','Rajasthani','BBQ','Afghani','Mediterranean','Konkan','Vietnamese','Hyderabadi','European']         



my_data = zomato_data.drop(columns=['address','online_order', 'votes','book_table',

       'rest_type', 'dish_liked', 'avg_cost', 'reviews_list',

       'type'],axis=1)



my_data.drop_duplicates(keep='last',inplace=True)



for l in my_list:

    print("\nCuisine Type:\t",l)

    

    temp_data = my_data[(my_data['cuisines'].str.contains(l)) & (my_data['rate'] > 4)]

    

    print(temp_data.drop(columns=['cuisines','rate'],axis=1).head(10))
zomato_data.drop(['address','name','rest_type','dish_liked','cuisines','reviews_list'],axis=1,inplace=True)
bin_edges = [0.0, 3.0, 4.0, 5.0]

bin_names = [1, 2, 3]



zomato_data['rest_class'] = pd.cut(zomato_data['rate'], bins=bin_edges, labels=bin_names,include_lowest=True)



y = zomato_data.loc[:,'rest_class']

X = zomato_data.drop(['rest_class','rate'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder



sc = StandardScaler()

le = LabelEncoder()

ohe = OneHotEncoder()



categorical_col = ['online_order','book_table' ,'type','locality']

continuous_col = ['votes','avg_cost']



scaled_col  =   sc.fit_transform(X_train[continuous_col])

encoded_col = X_train[categorical_col].apply(le.fit_transform)



X_train_processed    =   np.concatenate([scaled_col,encoded_col],axis=1)



scaled_col  =   sc.fit_transform(X_test[continuous_col])

encoded_col =  X_test[categorical_col].apply(le.fit_transform)

X_test_processed     =   np.concatenate([scaled_col,encoded_col],axis=1)



y_train = y_train.values

y_test = y_test.values

classifiers = {

    'Logistic Regression': LogisticRegression(),

    'Decision Tree': DecisionTreeClassifier(),

    'Random Forest': RandomForestClassifier(),

    'Support Vector Classifier': SVC(),

}



print("Accuracy of different models\n")

for key, model in classifiers.items():

    model.fit(X_train_processed, y_train)

    

    y_pred_test = model.predict(X_test_processed)

    acc_test = round(accuracy_score(y_test, y_pred_test) * 100,2)

    

    print(str(key) + ' : ' + str(acc_test) + '%')
