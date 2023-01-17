# imports

import pandas as pd

import seaborn as sb



#file load

df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')



df.head()
# ages in order to display in graph

age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']



# graph

graph = sb.violinplot(x='Age',y='Purchase',data=df,order=age_order, color='grey')



graph.axes.set_ylabel('Purchase value')



graph.axes.figure.set_figwidth(val=10)

graph.axes.figure.set_figheight(val=10)
# product purchase frequency

cProd = df['Product_ID'].value_counts()



# reseting the index

count = cProd.reset_index()



# graph

bars = sb.barplot(data=count.head(8),x='index',y='Product_ID', color='grey')

bars.axes.figure.set_figwidth(val=10)

bars.axes.set_xlabel('Products')

bars.axes.set_ylabel('Purchases')
# 5 most frequent occupations

occup = df['Occupation'].value_counts().head(5)



# frequency table to list 

occup = list(occup.reset_index()['index'])



occup_df = df[df['Occupation'].isin(occup)]



# graph

graph2 = sb.violinplot(x='Age',y='Purchase',data=occup_df, order=age_order, color='grey')



graph2.axes.set_ylabel('Purchase value')



graph2.axes.figure.set_figwidth(val=10)

graph2.axes.figure.set_figheight(val=10)
# purchases > 9000

df_9000 = df[df['Purchase'] > 9000]



# list of occupations

occup_order = df_9000['Occupation'].value_counts().reset_index()



# most frequent

occup_order = list(occup_order['index'])



# groupby and count by Occupation and Marital Status

grp = df_9000.groupby(by=['Occupation','Marital_Status']).count().reset_index()



# graph

graph3 = sb.barplot(x='Occupation',y='User_ID',hue='Marital_Status',data=grp,order=occup_order, color='grey')



graph3.axes.set_ylabel('Purchases with value over 9000')



# legend

for t, l in zip(graph3.axes.legend_.texts, ['Single','Married']): t.set_text(l)



graph3.axes.figure.set_figwidth(val=10)

graph3.axes.figure.set_figheight(val=10)