# data and libraries loading

import pandas as pd

import seaborn as sb



df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')



df.head()

# get the age order to display in the graph

age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']



# graph resizing and ploting

graph = sb.violinplot(x='Age',y='Purchase',data=df,order=age_order)



graph.axes.set_ylabel('Purchase value')



graph.axes.figure.set_figwidth(val=10)

graph.axes.figure.set_figheight(val=10)
# get the product purchase frequency

cProd = df['Product_ID'].value_counts()



# make the product column selectable by reseting the index wich is the product column already ordered by frequency

count = cProd.reset_index()



# graph resizing, relabeling and ploting

bars = sb.barplot(data=count.head(8),x='index',y='Product_ID')

bars.axes.figure.set_figwidth(val=10)

bars.axes.set_xlabel('Products')

bars.axes.set_ylabel('Purchases')

# get the top 5 frequent occupations

occup = df['Occupation'].value_counts().head(5)



# transform the frequency table of occupation into a list 

occup = list(occup.reset_index()['index'])



# make a new data frame with the correct purchases by filtering the occupation column with the occupation list

occup_df = df[df['Occupation'].isin(occup)]



# graph resizing, relabeling and ploting with the new dataframe (with only the 5 most frequent occupations)

graph2 = sb.violinplot(x='Age',y='Purchase',data=occup_df, order=age_order)



graph2.axes.set_ylabel('Purchase value')



graph2.axes.figure.set_figwidth(val=10)

graph2.axes.figure.set_figheight(val=10)

# filter the purchases with purchase value over 9000

df_9000 = df[df['Purchase'] > 9000]



# get the list of occupations of the new dataframe

occup_order = df_9000['Occupation'].value_counts().reset_index()



# transform the occupation frequency dataframe into a occupation list already ordered by most frequent

occup_order = list(occup_order['index'])



# make a group counting of the number of purchases by occupation and marital status of the new dataframe (filtered by purchase value)

# and index reseting to make another frequency dataframe, but with the grouped count information 

grp = df_9000.groupby(by=['Occupation','Marital_Status']).count().reset_index()



# graph resizing, relabeling and ploting with the grouped frequency dataframe

graph3 = sb.barplot(x='Occupation',y='User_ID',hue='Marital_Status',data=grp,order=occup_order)



graph3.axes.set_ylabel('Purchases with value over 9000')



# for each legend generated in hue category, set text ....

for t, l in zip(graph3.axes.legend_.texts, ['Single','Married']): t.set_text(l)



graph3.axes.figure.set_figwidth(val=10)

graph3.axes.figure.set_figheight(val=10)