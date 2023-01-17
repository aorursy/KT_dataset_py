
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/USvideos.csv')
df.info()

df.channel_title = df.channel_title.dropna()
df.info()
df.head(20)
#shows top 5 likes
df.nlargest(5, 'likes')
df.sort_values(by=["video_id","views"], ascending=False)
df.info()
#df = df.drop_duplicates("video_id")
df = df.drop_duplicates("video_id", keep="first")

#after df.drop_duplicates("video_id")
#reduce number of records from 24,951 to 4,881
df.info()
#shows top 5 likes
df.nlargest(5, 'likes')


#shows top 10 likes and views
#df.nlargest(10, ['likes','views'])
sns.pairplot(df)
#pairplot will display a scatter plot of  a combination values columns, not a categoical columns
#This will work on your local jupyter notebook, but not on Kaggle Kernel

sns.pairplot(df[["category_id","views","likes","dislikes"]], hue="category_id")
df["views"].corr(df["likes"])
#I suggest useing .corr to visualize the correlation matrix. To visualize the correlation matrix between features, you can create a "Heat Map" with seaborn. Or use this convenient pandas styling options is a viable built-in solution:

corr = df.corr()
print(type(corr))
corr.style.background_gradient()
#Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
#Cols = df[["category_id","views","likes","dislikes"]]
#df = DataFrame(, columns=Cols)

#sns.heatmap(df, annot=True)
#df.style.background_gradient(cmap='summer')
#select top 100 videos by top views 
df_top100views = df.nlargest(100, 'views')
df_top100views
sns.pairplot(df_top100views[["category_id","views","likes","dislikes"]], hue="category_id")

#merge in with the category_name from USvideo_categories.csv
df_category =pd.read_csv("../input/USvideo_categories.csv", header=None, names=["category_id","category_name"])
df_category
df = df.merge(df_category,on="category_id")
df
sns.pairplot(df[["category_name","views","likes","dislikes"]], hue="category_name")
#will the category_name also shows on the .corr correlation matrix? Why?
corr = df.corr()
corr.style.background_gradient()
