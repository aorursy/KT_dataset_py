import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

plt.style.use('ggplot')

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go
bf = pd.read_csv("../input/dataviz-facens-20182-ex3/BlackFriday.csv")

bf.head()
bf.describe()
classificacao = [["User_ID", "Qualitativa Nominal"],

            ["Product_ID","Qualitativa Nominal"],

            ["Gender","Qualitativa Nominal"],

            ['Age','Qualitativa Ordinal)'],

            ["Occupation","Qualitativa Nominal"],

            ["City_Category","Qualitativa Nominal"],

            ["Stay_In_Current_City_Years","Quantitativa Discreta"],

            ["Marital_Status","Qualitativa Nominal"],

            ["Product_Category_1","Qualitativa Nominal"],

            ["Product_Category_1","Qualitativa Nominal"],

            ["Product_Category_1","Qualitativa Nominal"],

            ["Purchase","Quantitativa Discreta"]]

classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])

classificacao
# isna

print(bf.isna().sum())
bf['Product_Category_2'].unique() 
bf['Product_Category_3'].unique() 
bf["Stay_In_Current_City_Years"].unique()

bf.fillna(0,inplace=True) 
bf['Gender'].unique() 
bf['Marital_Status'].unique() 
fig, eixos = plt.subplots(nrows=1, ncols=2, figsize=(15,7))

pie_1 = eixos[0].pie(bf['Gender'].value_counts(), explode=explode, labels=bf['Gender'].unique(), autopct='%1.1f%%')

eixos[0].set_title('Gender')

eixos[0].axis('equal')



pie_2 = eixos[1].pie(bf['Marital_Status'].value_counts(), explode=explode, labels=bf['Marital_Status'].unique(), autopct='%1.1f%%', startangle=90)

eixos[1].set_title('Marital Status')

plt.axis('equal')

plt.show()
data = bf.groupby(['Gender','Marital_Status'])['Gender'].count();

explode = (0.1, 0.1, 0.1, 0.1)

plt.figure(figsize=(10,5));

plt.pie(data.values,labels = data.index, explode=explode, autopct='%1.1f%%',shadow=True);

plt.title('Gender and Marital status');
bf.sort_values(by='Age', inplace=True, ascending=True)

plt.figure(figsize=(15,7))

sns.countplot(bf['Age'])

plt.title('Quantity by Age');
occupation = bf['Occupation'].value_counts();

plt.figure(figsize = (15,7));

plt.bar(occupation.index,occupation.values,color ='coral');

plt.xticks(occupation.index);

plt.xlabel('Occupation Types');

plt.ylabel('Count of people');

plt.title('Occupation Frequency');
occupation = bf['Occupation'];

purchase = bf['Purchase']

plt.figure(figsize = (15,7));

plt.figure(figsize=(15,7))

sns.barplot(x=occupation,y=purchase,data=bf,hue='Marital_Status',ci=0)

plt.xlabel('Occupation and Marital_Status');

plt.ylabel('Purchase ($)');

plt.title('Purchase by Occupation and Marital Status');

plt.show()
data = bf.groupby('Age')['Purchase'].mean()

data = pd.DataFrame({'Age':data.index, 'Average Purchase':data.values})



plt.figure(figsize=(15,7))

plt.plot('Age','Average Purchase','ys-',data = data, color=(1.0,0.2,0.3));

plt.grid();

plt.xlabel('Age');

plt.ylabel('Average Purchase ($)');

plt.title('Age range X Average amount spent');

avg_occupation = bf.groupby('Occupation')['Purchase'].mean();

plt.figure(figsize=(15,7));

plt.plot(avg_occupation.index, avg_occupation.values,'go-');

plt.xlabel('Occupation types');

plt.ylabel('Average purchase ($)');

plt.title('Average amount of purchase for each Occupation');

plt.xticks(avg_occupation.index);
data = bf.sort_values(by=['Gender'])

plt.figure(figsize=(15,6.5))    

sns.set_style("whitegrid")

sns.boxplot(x=data['Gender'], y=data['Purchase'], data=data) #, order=list(sorted_nb.index))

plt.title('Distribution of amounts spent by gender')

plt.xlabel('Gender')

plt.ylabel('Purchase ($)')

plt.show()
data = bf.sort_values(by=['Marital_Status'])

plt.figure(figsize=(15,6.5))    

sns.set_style("whitegrid")

sns.boxplot(x=data['Marital_Status'], y=data['Purchase'], data=data) #, order=list(sorted_nb.index))

plt.title('Distribution of amounts spent by marital status')

plt.xlabel('Marital Status')

plt.ylabel('Purchase ($)')

plt.show()
plt.figure(figsize=(10,6))

sns.heatmap(bf.corr(),annot=True,cmap ="YlGnBu", linewidths=0.1)