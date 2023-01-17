import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
auto=pd.read_csv('../input/Automobile_data.txt')
auto
automobile=auto.replace('?', np.nan)
automobile
automobile.isnull()
automobile
automobile.dtypes
automobile['bore']=pd.to_numeric(automobile['bore'],errors='coerce')
automobile['stroke']=pd.to_numeric(automobile['stroke'],errors='coerce')
automobile['horsepower']=pd.to_numeric(automobile['horsepower'],errors='coerce')
automobile['peak-rpm']=pd.to_numeric(automobile['peak-rpm'],errors='coerce')
automobile['price']=pd.to_numeric(automobile['price'],errors='coerce')
automobile.dtypes
automobile.make.value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('Make')
plt.ylabel('Count')
automobile['num-of-cylinders'].value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('num-of-cylinders')
plt.ylabel('Count')
automobile['fuel-system'].value_counts().plot(figsize=(15,8),kind='barh',stacked=True,colormap='bone')
plt.xlabel('Count')
plt.ylabel('fuel-system')
automobile['drive-wheels'].value_counts().plot(figsize=(15,8),kind='bar',stacked=True,colormap='bone')
plt.xlabel('drive-wheels')
plt.ylabel('Count')
import numpy as n
import matplotlib.pyplot as plt


# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Get the histogramp
Y,X = n.histogram(automobile['price'], range=(5000,23000),bins=5)
plt.xlabel("Price of Cars")
plt.ylabel("Frequency")
plt.title("Average Price of Cars")
plt.axvline(x=automobile['price'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()
import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('YlGnBu')

# Get the histogramp
Y,X = n.histogram(automobile['city-mpg'], range=(15,48),bins=5)
plt.xlabel("Number Of Miles in City")
plt.ylabel("Frequency")
plt.title("Average Miles Per Gallon in City")
plt.axvline(x=automobile['city-mpg'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()
import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlGn')

# Get the histogramp
Y,X = n.histogram(automobile['highway-mpg'], range=(19,48),bins=5)
plt.xlabel("Number Of Miles on Highway")
plt.ylabel("Frequency")
plt.title("Average Miles Per Gallon on Highway")
plt.axvline(x=automobile['highway-mpg'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()
automobile['bore_stroke_ratio'] = automobile['bore']/automobile['stroke']
automobile
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(automobile['bore_stroke_ratio'],automobile['horsepower'],color='blue')
plt.title('Permormance Criteria')
plt.xlabel('Bore : Stroke Ratio')
plt.ylabel('Horsepower')
plt.show()
sns.regplot(x='bore_stroke_ratio', y='horsepower', data=automobile)
oversquared = automobile[automobile['bore_stroke_ratio']>1]
import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('PuRd')

# Get the histogramp
Y,X = n.histogram(oversquared['bore_stroke_ratio'], range=(1,1.6),bins=5)
plt.xlabel("Oversquared Engines Range")
plt.ylabel("Frequency")
plt.title("Average of Oversquared Engines/Engines which are more Powerful")
plt.axvline(x=oversquared['bore_stroke_ratio'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()
oversquared.shape
undersquared = automobile[automobile['bore_stroke_ratio']<1]
import numpy as n
import matplotlib.pyplot as plt

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('GnBu')

# Get the histogramp
Y,X = n.histogram(undersquared['bore_stroke_ratio'], range=(0.85,0.99),bins=5)
plt.xlabel("Undersquared Engines Range")
plt.ylabel("Frequency")
plt.title("Average of Undersquared Engines/Engines which have more Torque")
plt.axvline(x=undersquared['bore_stroke_ratio'].mean(),linewidth=3,color='r')
x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C,width=X[1]-X[0])
plt.show()
undersquared.shape
automobile['oversquared']=automobile['bore_stroke_ratio']>1
automobile['undersquared']=automobile['bore_stroke_ratio']<1

