import seaborn as sns

from matplotlib import pyplot as plt

import pandas as pd
data = pd.read_csv("../input/vgsales.csv")
data.head()
data.info()
data.describe()
def categories_info(data) :

    print("----------")

    print("Categorical Informations")

    print("----------")

    for col in data.columns :

        if data[col].dtypes == 'object' and col != "Name" : #If you want to see "Name" category, delete 'and col != "Name"'

            print("")

            print("**********")

            print("Values for ", col)

            print("Number of different values : ", len(data[col].value_counts()))

            print("**********")

            print(data[col].value_counts())

            print(data[col].value_counts().describe())



categories_info(data)
#Let's affect this specific dataframe, as we will use it several times

year_null = data[data['Year'].isnull()]

year_null.head(10)
year_null.describe()
categories_info(year_null)
data_cleaned = data.dropna()

data_cleaned2 = data_cleaned[data_cleaned['Global_Sales'] > 1]

data_cleaned3 = data_cleaned[data_cleaned['Global_Sales'] < 10]

data_cleaned4 = data_cleaned2[data_cleaned2['Global_Sales'] < 10]



#Games sold between 1 and 10 Million by publisher for more than 10 games

mask = data_cleaned4['Publisher'].value_counts().sort_values() > 10

long = mask[mask.values==True]

clean = data_cleaned4[data_cleaned4['Publisher'].isin(long.index.values)]



#Games sold between 1 and 10 Million

mask = data_cleaned3['Publisher'].value_counts().sort_values() > 10

long = mask[mask.values==True]

clean2 = data_cleaned3[data_cleaned3['Publisher'].isin(long.index.values)]
print(len(data_cleaned2))

print(len(data_cleaned4))
plot_col = ['Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']



def plot_distrib(data, data2) :

    

    #Size the figure that will hold all the subplots

    plt.figure(figsize=(12, 15))

    i = 1

    

    #For all the columns, make a subplot

    for col in plot_col :

        plt.subplot(int("32" + str(i)))

        

        #If this is a numerical type : 

        if data[col].dtypes != object :

            

            #Plot the values

            maxi = data[col].value_counts().sort_values(ascending=False).index[0]

            maxi2 = data2[col].value_counts().sort_values(ascending=False).index[0]

            sns.distplot(data[col], kde=False, label="All games")

            sns.distplot(data2[col], kde=False, label="Sales > 1 M")

            

            #Add information on the most popular and the most successfull parameters

            plt.title("{} = popular {}, {} = successfull".format(maxi, col, maxi2))

            plt.legend()

            

        #If this is a categorical type :

        else :

            

            #Restrain from having too many values on the sample plot. Here only the values that occured more than 100 times

            mask = data[col].value_counts().sort_values() > 100

            long = mask[mask.values==True]

            clean = data[data[col].isin(long.index.values)]

            

            #Plot the values and cleaned values, in descending order of occurence

            maxi = clean[col].value_counts().sort_values(ascending=False).index

            maxi2 = data2[col].value_counts().sort_values(ascending=False).index

            sns.countplot(clean[col], order=maxi, label="All games")

            sns.countplot(data2[col], order=maxi, label="Sales > 1 M", saturation=0.4)

            

            #Add information on the most popular and the most successfull parameters

            plt.title("{} = popular {}, {} = successfull".format(maxi[0], col, maxi2[0]))

            plt.legend()

            

        #Rotate the labels on x-axis

        plt.xticks(rotation=90)

        i += 1

    

    #Espace each subplot to avoid overlap

    plt.subplots_adjust(hspace=0.5)





plot_distrib(clean2, clean)

plt.show()
def plot_cross_num(data2) :

    

    #Search for the numerical types

    col_num = []

    for col in plot_col :

        if data2[col].dtypes != object :

            col_num.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(12, 15))

    i = 1

    

    #For each column

    for col in col_num :

        col_num.remove(col)

        for col2 in col_num :

        

            #Plot the values

            plt.subplot(int("32" + str(i)))

            sns.lmplot(x=col, y=col2, data=data2, fit_reg=False, hue="Genre", palette="Set1")            

            sns.kdeplot(data2[col], data2[col2], n_levels=20)



            #Add information

            plt.title("{} co-plotted with {}".format(col, col2))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

        

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)

    

    

plot_cross_num(clean)

plt.show()
def plot_cross_cat(data2) :

    

    #Search for the categorical types

    col_cat = []

    for col in plot_col :

        if data2[col].dtypes == object :

            col_cat.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(15, 30))

    i = 1

    

    #For each column

    for col in col_cat :

        col_cat.remove(col)

        for col2 in col_cat :

        

            #Plot the values

            plt.subplot(int("42" + str(i)))

            table_count = pd.pivot_table(data2,values=['Global_Sales'],index=[col],columns=[col2],aggfunc='count',margins=False)

            sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)



            #Add information

            plt.title("{} co-plotted with {}".format(col, col2))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

        

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)

    

    

plot_cross_cat(clean)

plt.show()
def plot_cross_cross(data2) :

    

    #Search for the types

    col_cat = []

    col_num = []

    for col in plot_col :

        if data2[col].dtypes == object :

            col_cat.append(col)

        else :

            col_num.append(col)

    

    #Size the figure that will contain the subplots

    plt.figure(figsize=(12, 15))

    i = 1

    j = len(col_cat)

    k = len(col_num)

    #For each column

    for cat in col_cat :

        for num in col_num :

        

            #Plot the values

            plt.subplot(int(str(j) + str(k) + str(i)))

            #mask = data[cat].value_counts().sort_values() > 100

            #long = mask[mask.values==True]

            #clean = data[data[cat].isin(long.index.values)]

            sns.violinplot(x=cat , y=num, data=data2, inner=None) 

            sns.swarmplot(x=cat, y=num, data=data2, alpha=0.7) 

            #table_count = pd.pivot_table(data2,values=['Global_Sales'],index=[num],columns=[cat],aggfunc='count',margins=False)

            #sns.heatmap(table_count['Global_Sales'],linewidths=.5,annot=True,fmt='2.0f',vmin=0)



            #Add information

            plt.title("{} co-plotted with {}".format(cat, num))

        

            #Rotate the x-label

            plt.xticks(rotation=90)

            i += 1

        

    #Adjust the subplots so that they don't overlap

    plt.subplots_adjust(hspace=0.5)

    

    

plot_cross_cross(clean)

plt.show()