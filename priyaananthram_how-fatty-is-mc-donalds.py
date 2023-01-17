# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df =pd.read_csv("../input/menu.csv")

print(df.shape)

df.head()

# Any results you write to the current directory are saved as output.
%matplotlib inline
df.groupby('Category')['Item'].count().plot(kind='bar')
import numpy as np

pd.pivot_table(df,index=['Category'],values=['Total Fat (% Daily Value)'],aggfunc=np.max).plot(kind='bar')
grp_by_df=df.groupby(['Category'],as_index=False)['Total Fat (% Daily Value)','Trans Fat','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)'].agg(np.max)

grp_by_df.columns=['Category','Max_Fat','Max_Trans_Fat','Max_Sat_Fat','Max_Cholestrol']



grp_by_df
df=df.merge(grp_by_df,left_on=['Category'],right_on=['Category'])

fatty_df=df.loc[df['Total Fat (% Daily Value)']==df.Max_Fat,['Category','Item','Total Fat (% Daily Value)','Cholesterol (% Daily Value)']]

fatty_df
fatty_df=df.loc[df['Trans Fat']==df.Max_Trans_Fat,['Category','Item','Total Fat (% Daily Value)','Trans Fat','Saturated Fat (% Daily Value)','Cholesterol (% Daily Value)']]

fatty_df.sort_values(by='Trans Fat',ascending=False)[0:19]
g=df.sort_values(by=['Total Fat (% Daily Value)','Cholesterol (% Daily Value)'])

g.loc[:,['Category','Item','Total Fat (% Daily Value)','Cholesterol (% Daily Value)']][0:38]
df['isGrilled']=df.Item.str.contains("Grilled")

df['hasEggWhites']=df.Item.str.contains("Egg Whites")



crispy_df1=df.loc[df.isGrilled==True,'Item'].str.replace('Grilled','Crispy')

crispy_df=df.loc[df.Item.isin(crispy_df1),['Item','Total Fat (% Daily Value)','Calories']]

grilled_df=df.loc[df.isGrilled==True,['Item','Total Fat (% Daily Value)','Calories']]



df1=grilled_df.reset_index(drop=True).merge(crispy_df.reset_index(drop=True),left_index=True,right_index=True)

df1.columns=['Items-Grilled','TotalFat-Grilled','Calories-Grilled','Items-Crispy','TotalFat-Crispy','Calories-Crispy']

df1=df1.drop('Items-Crispy',axis=1)

df1['Item']=df1['Items-Grilled'].str.replace("Grilled","")

df1=df1.drop('Items-Grilled',axis=1)

df1.index=df1.Item

from pylab import rcParams

rcParams['figure.figsize'] = 8, 10

df1.loc[:,['TotalFat-Grilled','TotalFat-Crispy','Item']].plot(kind='barh',title="Fat grilled versus crispy")
df1.loc[:,['Calories-Grilled','Calories-Crispy','Item']].plot(kind='barh',title="Fat grilled versus crispy")