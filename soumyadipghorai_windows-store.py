import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from wordcloud import WordCloud
sns.set(color_codes = True)
missing_values = ['NaN', 'Free'] #set the NaN values 
df = pd.read_csv('../input/windows-store/msft.csv', na_values = missing_values) #considering free as NaN
df
df.describe()
df.info()
df.Date = pd.to_datetime(df.Date) #changed it from object 
df.drop(index=5321, inplace = True) # dropped the row
# as this row has almost all NaN values
df.fillna(value = 0, inplace = True) # now only the NaNs are left in the price col 
# so 1st we made them 0 from NaN
df['Price'] = df['Price'].replace(['₹',',','<'], '', regex=True) #Now we removed the symbols used in price col 
# now the dataframe is ready to go 
df_rate = df[['Price', 'No of people Rated']]  # selecting dataframe 
df_rate = df_rate.apply(pd.to_numeric)    # prices were ojects 
# So making them numeric
df_rate = df_rate.sort_values('Price', ascending = False); # sorting by price 
fig = plt.gcf();
fig.set_size_inches(15,7);
sns.swarmplot(x= 'Price', y='No of people Rated', data = df_rate).set_title('Prdocut distribution across the price'); #plotted
plt.xticks(rotation=90);

tatal_rated = df[['No of people Rated', 'Category']].groupby('Category').sum()
# grouped by category
fig =plt.gcf();
fig.set_size_inches(15,7);
sns.barplot(x=tatal_rated.index, y='No of people Rated', data= tatal_rated); #plot
plt.xticks(rotation=90);
# which product has got the highest average rating 
df_mean_rating = df[['Rating', 'Category']].groupby('Category').mean()
# grouped by categories 
df_mean_rating = df_mean_rating.sort_values('Rating', ascending = False)
# sorted the values 
fig =plt.gcf();
fig.set_size_inches(10,7);
plt.bar(df_mean_rating.index, 'Rating', 
        data= df_mean_rating, color ='#00ed14');
# plotted using color green
plt.xticks(rotation=90);
# most rated rating 
total_rating = df.groupby('Rating').sum()
fig=plt.gcf();
fig.set_size_inches(20,7);
sns.barplot(x=total_rating.index, y='No of people Rated', data=total_rating );
plt.xlabel('Rating');
plt.ylabel('No of people Rated');
# total rating in each year so far 
years = df[['Date', 'No of people Rated']].groupby(df['Date'].dt.year).sum()
# grouped by dates 
fig=plt.gcf();
fig.set_size_inches(15,7);
sns.barplot(x=years.index,y='No of people Rated', data=years);
# plotted 
plt.xlabel('Date');
plt.ylabel('No of people Rated');
# which category has how many products 
cat_total = df.groupby('Category').count()
cat_total = cat_total.sort_values('Name', ascending = False)
# sorted 
fig=plt.gcf();
fig.set_size_inches(15,7);
sns.barplot(x=cat_total.index,y='Name', data=cat_total);
plt.xticks(rotation=80);
# rotated the names along the x axis 
plt.xlabel('Category Name');
plt.ylabel('Total Itmes');
# gave labels 
# best free apps in category 
my_color= ["#fa4a05", "#00ed14", "#f7f420", "#21abeb"]
def best(n): # defining the function
    free_product = df[df.Price == 0.0]   # data of only the free products 
    best_free_prodcut = free_product[free_product.Rating == 5.0] #only those who have got 5.0 rating 
    best_free = best_free_prodcut[best_free_prodcut.Category == n ]   #select the category 
    constant = best_free['No of people Rated'].median() + best_free['No of people Rated'].std()  # selected the constant 
# here we used a statistical measure 
# we have taken the sum of median and standard deviation to find only the top products 
# but it is unnecessary as we are selecting only the top 4
    best = best_free[best_free['No of people Rated'] > constant ]  # so we have selected the toppers of the category 
    top =best.sort_values(by = 'No of people Rated', ascending = False)
#sorted them according to the total no of ratings 
    top_ =top.head(4)
#selected only the top 4 
    fig=plt.gcf();
    fig.set_size_inches(7,4);
    x = sns.barplot(x=top_.Name, y='No of people Rated', data=top_ , 
                    palette = my_color).set_title(n); #plotted with the four colors 
# given title 
    plt.xticks(rotation =80);
    plt.xlabel('Top 4 Free Apps');  #given labels x aixs 
    plt.ylabel('No of people Rated'); #given labels y axis 
    return x
# returning the plot 
best('News and Weather');
best('Books');
wordCloud = WordCloud(background_color='white',max_font_size = 50).generate(' '.join(df.Category))
plt.figure(figsize=(10,6))
plt.axis('off')
plt.imshow(wordCloud)
plt.show()