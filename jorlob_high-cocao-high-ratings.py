import pandas as pd



import matplotlib.pyplot as plt

plt.style.use('seaborn')
df = pd.read_csv('../input/flavors_of_cacao.csv',  na_values = [" ",'Â '])
df.columns
df.columns = ['Company', 'Specific_Bean_Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Company_Location', 'Rating', 'Bean_Type', 'Broad_Bean_Org']
df.info()
df.head()
df['Company'] = df['Company'].astype('category')

df['Review_Date'] = pd.to_numeric(df['Review_Date'])

df['Cocoa_Percent']= df['Cocoa_Percent'].str.strip('%').astype(float)
df.describe()
df['Rating'].plot(kind = 'hist')

plt.title('Histogram of Ratings')

plt.xlabel('Rating')

plt.ylabel('Count of Ratings')

plt.show()



df['Cocoa_Percent'].plot(kind = 'hist')

plt.title('Histogram of Cocoa_Percent')

plt.xlabel('Cocoa_Percent')

plt.ylabel('Count')

plt.show()
df.plot(kind = 'scatter', x = 'Review_Date', y = 'Rating', rot=70)

plt.xlabel('Year of Review')

plt.title('Scatterplot of Review Year and Rating')

plt.show()
df.plot(kind = 'scatter', x = 'Rating', y = 'Cocoa_Percent', rot=70)

plt.ylabel('Cacoa %')

plt.title('Scatterplot of Rating and Cacoa %')

plt.show()