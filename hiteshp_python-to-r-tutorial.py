# Both langugares have library to perform set of functions.

# In Python

# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np   # linear algebra
import matplotlib.pyplot as plt  #For ploting graphs
import seaborn as sns #For ploting graphs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

# In R

# library(readr) # CSV file I/O, e.g. the read_csv function
# library(ggplot2) # Data visualization
# library(dplyr) # Data Manupulation
#Creating a DataFrame by passing a dict of objects that can be converted to series-like.

# In Python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df

#In R

# dates <-seq(as.Date("2013/01/01"), by = "day", length.out = 6)
# df<- data.frame(date=dates, 
#                 A=runif(6) , 
#                 B=runif(6) , 
#                 C=runif(6) , 
#                 D=runif(6)  
#                )
# df



# In Python
df2 = pd.DataFrame({ 'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })
df2


# dates <-as.Date("2013/01/01")

# df2<- data.frame(date=dates, 
#                 A=1 , 
#                 B=runif(4) , 
#                 C=runif(4) , 
#                 D=runif(4) , 
#                 E=c("test","train","test","train") , 
#                 F="foo"
#              )
# df2
#data types
# In Python
df2.dtypes

# In R
# class(df)

#See the top & bottom rows of the frame
# In Python
df.head()

# In R
# head(df)
# In Python
df.tail(3)

# In R
# tail(df,3)
# Get dimension of data frame 
# In Python
df.shape

# In R
# dim(df)
#Get number of rows
# In Python
df.shape[0]

# In Python
# nrow(df)
#Get number of columns
# In Python
df.shape[1]


# In R
# ncol(df)
# In Python
df.columns

# In R
# colnames(df)
#Describe shows a quick statistic summary of your data
# In Python
df.describe()

# In R
# summary(df)
#Transposing your data
# In Python
df.T

# In R
# transpose(df)
#Sorting by values
# In Python
df.sort_values(by='B')

# In R
# df[order(df$B),]
#Selecting a single column, which yields a Series, equivalent to df.A
# In Python
df['A']

# In R
# df[,A]
#Selecting via [], which slices the rows.
#In python row ID starts from 0, where as in R row ID starts from 1
# In Python
df[0:3]

# In R
# df[c(1:3),]
# In Python
df['20130102':'20130104']


# In R
# df[between(df$date, 20130102, 20130104), ]

#Selecting on a multi-axis by label
# In Python
df.loc[:,['A','B']]

# In R
# df[,c('A','B')]
#Showing label slicing, both endpoints are included
# In Python
df.loc['20130102':'20130104',['A','B']]

# In R
# df[,c('A','B')]
#Reduction in the dimensions of the returned object
# In Python
df.loc['20130102',['A','B']]

# In R
# df[c(1:3),]
#For getting a scalar value
# In Python
df.loc[dates[0],'A']

# In R
# df[c(1:3),]
#For getting fast access to a scalar (equiv to the prior method)
# In Python
df.at[dates[0],'A']

# In R
# df[c(1:3),]
#Select via the position of the passed integers
# In Python
df.iloc[3]

# In R
# df[c(1:3),]
#By integer slices, acting similar to numpy/python
# In Python
df.iloc[3:5,0:2]

# In R
# df[c(1:3),]
#By lists of integer position locations, similar to the numpy/python style
# In Python
df.iloc[[1,2,4],[0,2]]

# In R
# df[c(1:3),]
#For slicing rows explicitly
# In Python
df.iloc[1:3,:]

# In R
# df[c(1:3),]
#For slicing columns explicitly
# In Python
df.iloc[:,1:3]

# In R
# df[c(1:3),]
#For getting a value explicitly
# In Python
df.iloc[1,1]

# In R
# df[c(1:3),]
#For getting fast access to a scalar (equiv to the prior method)
# In Python
df.iat[1,1]

# In R
# df[c(1:3),]
#Boolean Indexing
# In Python
df[df.A > 0]

# In R
# df[c(1:3),]
#Selecting values from a DataFrame where a boolean condition is met.
# In Python
df[df > 0]

# In R
# df[c(1:3),]
#Using the isin() method for filtering:
# In Python
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2


# In R
# df2 <- df
# df2$E <- c('one', 'one','two','three','four','three')
# df2
#Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.
# In Python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
df1


# In R
# dates <-seq(as.Date("2013/01/01"), by = "day", length.out = 6)
# df1 <- data.frame(date=dates[1:4])

                 
# df1 <- cbind(df1,df[1:4,c('A','B','C','D')])
# df1$E<-1
# df1[3:4,'E']<- NA
# df1
#To drop any rows that have missing data.
# In Python
df1.dropna(how='any')

# In R
#na.omit(df1)
#Filling missing data
# In Python
df1.fillna(value=5)

# In R
# df2<-df1
# df1[is.na(df1)] <- 5
# df1
#To get the boolean mask where values are nan
# In Python
pd.isnull(df1)

# In R
#is.na(df2)
# In Python
df.mean()

# In R
#sapply(df[,-1], mean, na.rm = T)
#Same operation on the other axis
# In Python
df.mean(1)

# In R
#data.frame(ID=df[,1], Means=rowMeans(df[,-1]))
# In Python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

s.str.lower()

# In R
# s <- c('A', 'B', 'C', 'Aaba', 'Baca', NA , 'CABA', 'dog', 'cat')
# tolower(s)
#Concat
# In Python
df = pd.DataFrame(np.random.randn(10, 4))
df

# In R
# df<- data.frame( 
#                 A=runif(10) , 
#                 B=runif(10) , 
#                 C=runif(10) , 
#                 D=runif(10)  
#                )
# df
# break it into pieces
# In Python
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# In R
# a<-df[1:3,]
# b<-df[3:7,]
# c<-df[7:10,]
# rbind(a,b,c)

#SQL style merges
# In Python
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

# In R
# left<-data.frame(key='foo',
#                 lval=c(1,2))
# right<-data.frame(key='foo',
#                 lval=c(4,5))
left

right
# In Python
pd.merge(left, right, on='key')

# In R
#merge(x = left, y = right, by = "key", all = TRUE)
#Another example that can be given is:
# In Python
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

# In R
# left<-data.frame(key=c('foo', 'bar'),
#                 lval=c(1,2))
# right<-data.frame(key=c('foo', 'bar'),
#                 lval=c(4,5))
left

right
# In Python
pd.merge(left, right, on='key')

# In R
#merge(x = left, y = right, by = "key", all = TRUE)

# In Python
data = pd.read_csv('../input/pokemon.csv')

# In R
# data <- read.csv('../input/pokemon.csv')

# In Python
data.head(10)

# In R
# head(data,10)
# In Python
data.info()

# In R
# str(data)
# In Python
data.columns

# In R
# colnames(data)
# In Python
series = data['Defense']        # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']]  # data[['Defense']] = data frame
print(type(data_frame))

# In R
# series <-data[,c('Defense')]
# print(class(series))


# 1 - Filtering Pandas data frame
# In Python
x = data['Defense']>200     # There are only 3 pokemons who have higher defense value than 200
data[x]

# In R
# x = data[data$Defense>200,]
# x
# library(dplyr)
# # with dplyr package
# data %>% filter(Defense>200)
# x
# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
# In Python
data[np.logical_and(data['Defense']>200, data['Attack']>100 )]



# In R
# x = data[data$Defense>200 & 'Attack'>100 ,]
# x

# # with dplyr package
# data %>% filter(Defense>200 & 'Attack'>100)


# This is also same with previous code line. Therefore we can also use '&' for filtering.
# In Python
data[(data['Defense']>200) & (data['Attack']>100)]

# In R
# x = data[data$Defense>200 & 'Attack'>100 ,]
# x

# # with dplyr package
# data %>% filter(Defense>200 & 'Attack'>100)



# For example lets look frequency of pokemom types
# In Python
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon

# In R
# colSums(!is.na(data$Type1))

# For example max HP is 255 or min defense is 5
# In Python
data.describe() #ignore null entries

# In R
# summary(data)


# Firstly lets create 2 data frame
# In Python
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row

# In R
# data1 = head(data)
# data2= tail(data)
# conc_data_row <- merge(data1,data2) # rbind(data1,data2)
# conc_data_row


# In Python
data1 = data['Attack'].head()
data2= data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col

# In R

# data1 = head(data$Attack)
# data2= head(data$Defense)
# conc_data_col = cbind(data1,data2)
# conc_data_col


# read data
# In Python
data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")
data.head()

# In R
# data <- read.csv('../input/pokemon.csv')

# indexing using square brackets
# In Python
data["HP"][1]
# using column attribute and row label
#data.HP[1]


# In R
# data[1,c("HP")]

# using loc accessor
# In Python
data.loc[1,["HP"]]

# In R

# data[1,c("HP")]

# Selecting only some columns
# In Python
data[["HP","Attack"]]

# In R
# data[,c("HP","Attack")]

# Slicing and indexing series
# In Python
data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive

# In R
# data[1:10,c("HP","Attack")]

# From something to end
# In Python
data.loc[1:10,"Speed":] 

# In R
# data[1:10,9:] 

# Creating boolean series
# In Python
boolean = data.HP > 200
data[boolean]

# In R
# boolean = data$HP > 200
# data[boolean]

# #With dplyr package
# data %>% filter(HP > 200)


# Combining filters
# In Python
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]

# In R
# first_filter = data$HP > 150
# second_filter = data$Speed > 35
# data[first_filter & second_filter]


# #With dplyr package
# data %>% filter(data$HP > 150 & data$Speed > 35 )

# Filtering column based others
# In Python
data.HP[data.Speed<15]

# In R
# data %>% filter(data$Speed<15 )
# Defining column using other columns
# In Python
data["total_power"] = data.Attack + data.Defense
data.head()

# In R
# data$total_power = data$Attack + data$Defense
# head(data)

#correlation map
# In Python
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# In R
# library(corrplot)
# # corrplot 0.84 loaded
# M <- cor(data)
# corrplot.mixed(M)

# Line Plot
# In Python
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot


# #In R
# ggplot(data=data, aes(x=Speed, y=len, group=1)) +
# geom_line(aes(y=Speed),col="red")+
# geom_line(aes(y=Defense),col="green")


# Scatter Plot 
# In Python
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot


# #In R
# ggplot(data=data, aes(x=Attack, y=Defense, group=1)) +
# geom_point() +
# labs(title = "Attack Defense Scatter Plot")


# Histogram
# In Python
# bins = number of bar in figure
data.Speed.plot(kind = 'hist',bins = 50,figsize = (15,15))

# #In R
# hist(data$Speed, breaks=50, col="blue")
# Box plots: visualize basic statistics like outliers, min/max or quantiles
# In Python
data.boxplot()

# #In R
# boxplot(data, las = 2)
# hist(data$Speed, breaks=50, col="blue")
# Plotting all data 
# In Python
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# it is confusing

# subplots
data1.plot(subplots = True)


# 
# par(1,2)
# #In R
# a<-ggplot(data=data, aes()) +
# geom_line(aes(y=Attack),col="red")+
# geom_line(aes(y=Defense),col="green")

# a<-ggplot(data=data, aes()) +
# geom_line(aes(y=Speed),col="red")

# a<-ggplot(data=data, aes()) +
# geom_line(aes(y=Defense),col="green")


# grid.arrange(a,b,c nrow=3,ncol=1)