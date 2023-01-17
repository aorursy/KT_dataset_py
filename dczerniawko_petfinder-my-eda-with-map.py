import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from wordcloud import WordCloud



import folium



import os

import cv2



%matplotlib inline

pd.set_option('max_columns', 30)
# DIY



def missing_values(df):

    for column in df.columns:

        null_rows = df[column].isnull()

        if null_rows.any() == True:

            print('%s: %d nulls' % (column, null_rows.sum()))

            

def cicle(df):    

    plt.figure(figsize=(10,7))

    names= 'Dog', 'Cat'

    size=df['Type'].value_counts()

    my_circle=plt.Circle((0,0), 0.7, color='white')

    plt.pie(size, labels=names, colors=['skyblue','red'])

    p=plt.gcf()

    p.gca().add_artist(my_circle)

    plt.title('Type of pets distribution', fontsize=15)

    plt.show()

    

def buzz_name(txt):

    wordcloud = WordCloud(width=480, height=480, max_font_size=50, min_font_size=10).generate(dog_txt)

    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.margins(x=0, y=0)

    plt.show()

    

def show_rand_img():    

    plt.rc('axes', grid = True)

    _, ax = plt.subplots(1, 3, figsize=(20, 20))

    images_train = os.listdir("../input/train_images/")

    random_img = np.random.randint(0, len(images_train) - 3)



    for i , file in enumerate(images_train[random_img:random_img + 3]):

        img = cv2.imread('../input/train_images/{}'.format(file))

        ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
df_train = pd.read_csv('../input/train/train.csv')

df_breed = pd.read_csv('../input/breed_labels.csv')

df_color = pd.read_csv('../input/color_labels.csv')

df_state = pd.read_csv('../input/state_labels.csv')
df_train.sample(3)
df_train.shape
df_breed.sample(5)
df_color
df_state
df_train.info()
df_train.describe()
df_train.isnull().any().any()
missing_values(df_train)
plt.rcParams['figure.figsize']=(18,10)

sns.heatmap(df_train.corr(), annot=True, linewidths=.5, fmt = ".2f", cmap="YlGnBu");
# target variable

ax = sns.countplot(x = 'AdoptionSpeed', data = df_train, palette = 'hls');

ax.set_title(label='Count of adoption speed', fontsize=20);
ax = sns.kdeplot(df_train['Age'], shade=True);

ax.set_title(label='Count of pets age', fontsize=20);
ax = sns.countplot(x = 'Gender', data = df_train, palette = 'hls');

ax.set_title(label='Count of adoption speed per gender', fontsize=20);
ax = sns.countplot(x="Color1", data=df_train, hue="AdoptionSpeed")

ax.set_title(label='Count of color and adoption speed', fontsize=20);
ax = sns.countplot(x="FurLength", data=df_train, hue="AdoptionSpeed")

ax.set_title(label='Count of fur lenght and adoption speed', fontsize=20);
ax = sns.countplot(x="MaturitySize", data=df_train, hue="AdoptionSpeed")

ax.set_title(label='Count of maturity size and adoption speed', fontsize=20);
ax = sns.countplot(x="Health", data=df_train, hue="AdoptionSpeed")

ax.set_title(label='Count of health and adoption speed', fontsize=20);
ax = sns.countplot(x="Sterilized", data=df_train, hue="AdoptionSpeed")

ax.set_title(label='Count of sterilized and adoption speed', fontsize=20);
# we have two type of animals - dog(1) and cat(2)

df_train['Type'].value_counts()
cicle(df_train)
ax = sns.countplot(x='Type',hue='Gender',data=df_train);

ax.set_title(label='Sex of the animal', fontsize=15);
ax = sns.violinplot(x='MaturitySize', y='Age',

                     hue='AdoptionSpeed',

                     data=df_train)

ax.set_title(label='Relation between maturity size and age with respected adoption speed', fontsize=20);
plt.figure(figsize=(10,7))

ax = sns.violinplot(x='Gender', y='Age',

                     hue='AdoptionSpeed',

                     data=df_train)

ax.set_title(label='Relation between gender and age with respected adoption speed', fontsize=20);
plt.figure(figsize=(10,7))

ax = sns.violinplot(x='FurLength', y='Age',

                     hue='AdoptionSpeed',

                     data=df_train)

ax.set_title(label='Relation between fur length and age with respected adoption speed', fontsize=20);
plt.title('Adoption time:')

ax = sns.countplot(x='Type',hue='AdoptionSpeed',data=df_train);

ax.set_title(label='Relation between adoption time and type', fontsize=20);
top10_names = df_train['Name'].value_counts().head(10)

top10_names.plot(kind='bar', title = 'Top 10 pet names');
sns.boxenplot(y='Age',x='AdoptionSpeed',hue='Type',data=df_train);
df_train[df_train['Age'] == df_train['Age'].max()]
ax = sns.violinplot(y='Fee',x='AdoptionSpeed',hue='Type',data=df_train);

ax.set_title(label='The most expensive fee', fontsize=20);
ax = sns.pointplot(x = 'Sterilized', y = 'AdoptionSpeed', hue = 'Health', data = df_train);

ax.set_title(label='Adoption speed vs Health and Sterilized', fontsize=20);
df_train['AdoptionSpeed'].value_counts().sort_index().plot('barh');

plt.title('Adoption speed classes counts');
plt.rcParams['figure.figsize']=(15,8)

dog_txt = ' '.join(df_train.loc[df_train['Type'] == 1, 'Name'].fillna('').values)

buzz_name(dog_txt)
cat_txt = ' '.join(df_train.loc[df_train['Type'] == 2, 'Name'].fillna('').values)

buzz_name(cat_txt)
show_rand_img()
data = pd.DataFrame({

'lat':[103.733333, 100.362778, 102, 101.692222, 115.219033, 102.251111, 102.25, 102.5, 101, 100.25, 100.3292, 117, 113.781111, 101.5, 103],

'lon':[1.483333, 6.128333, 5.25, 3.153889, 5.315894, 2.188889, 2.75, 3.75, 4.75, 6.5, 5.4145, 5.25, 3.038056, 3.333333, 4.75],

'name':['Johor', 'Kedah', 'Kelantan', 'Kuala Lumpur', 'Labuan', 'Malakka', 'Negeri Sembilan', 'Pahang', 'Perak', 'Perlis', 'Penang', 'Sabah', 'Sarawak', 'Selangor', 'Terengganu']

})

data

 

m = folium.Map(location=[5, 108], tiles="Mapbox Bright", zoom_start=6)

 

for i in range(0,len(data)):

    folium.Marker([data.iloc[i]['lon'], data.iloc[i]['lat']], popup=data.iloc[i]['name']).add_to(m)

    

display(m)
# df_train = df_train[df_train['Description'].notnull()]