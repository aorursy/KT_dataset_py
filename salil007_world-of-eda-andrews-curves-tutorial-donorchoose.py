import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
donations.head()
donations.shape
donors.head()
donors.shape
schools.head()
schools.shape
teachers.head()
teachers.shape
projects.head()
projects.shape
resources.head()
resources.shape
donations.groupby('Donation Included Optional Donation')['Donation Included Optional Donation'].count().plot.bar()
temp = donations['Donation Included Optional Donation'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
donations['Donation Amount'].value_counts().head()
#TOP 5 occuring Donation Amounts
temp = donations['Donation Amount'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
donations['Donor Cart Sequence'].value_counts().head()
#TOP 5 occuring Donor Cart Sequence
temp = donations['Donor Cart Sequence'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
donors.groupby('Donor Is Teacher')['Donor Is Teacher'].count().plot.bar()
temp = donors['Donor Is Teacher'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 10 occuring Cities
temp = donors['Donor City'].value_counts()[:10]
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 10 occuring States
temp = donors['Donor State'].value_counts()[:10]
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 10 occuring Zips
temp = donors['Donor Zip'].value_counts()[:10]
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
schools.groupby('School Metro Type')['School Metro Type'].count().plot.bar()
temp = schools['School Metro Type'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School Percentages Free Lunch
temp = schools['School Percentage Free Lunch'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School States
temp = schools['School State'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School Zips
temp = schools['School Zip'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School Cities
temp = schools['School City'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School Counties
temp = schools['School County'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring School Districts
temp = schools['School District'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
teachers.groupby('Teacher Prefix')['Teacher Prefix'].count().plot.bar()
temp = teachers['Teacher Prefix'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['Year'] = teachers['Teacher First Project Posted Date'].dt.year
teachers['Month'] = teachers['Teacher First Project Posted Date'].dt.month
teachers['Day'] = teachers['Teacher First Project Posted Date'].dt.day
df1 = teachers['Year'].value_counts()
df2 = teachers['Month'].value_counts()
df3 = teachers['Day'].value_counts()
#TOP 5 occuring Years
df1.head().plot.bar()
temp = teachers['Year'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring Months
df2.head().plot.bar()
temp = teachers['Month'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring Days
df3.head().plot.bar()
temp = teachers['Day'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring Teacher Project Posted Sequences
temp = projects['Teacher Project Posted Sequence'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
projects.groupby('Project Type')['Project Type'].count().plot.bar()
temp = projects['Project Type'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                     ).generate(str(projects['Project Title'].tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Project Title", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                     ).generate(str(projects['Project Essay'].sample(2000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Project Essay", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#TOP 5 occuring Project Subject Category Tree
temp = projects['Project Subject Category Tree'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring Project Subject Subcategory Tree
temp = projects['Project Subject Subcategory Tree'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
projects.groupby('Project Grade Level Category')['Project Grade Level Category'].count().plot.bar()
temp = projects['Project Grade Level Category'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
projects.groupby('Project Resource Category')['Project Resource Category'].count().plot.bar()
temp = projects['Project Resource Category'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                     ).generate(str(resources['Resource Item Name'].sample(2000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Resource Item Name", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud( max_font_size=50, 
                       stopwords=STOPWORDS,
                       background_color='black',
                     ).generate(str(resources['Resource Vendor Name'].sample(4000).tolist()))

plt.figure(figsize=(14,7))
plt.title("Wordcloud for Top Keywords in Resource Vendor Name", fontsize=35)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#TOP 5 occuring Resource Quantites
temp = resources['Resource Quantity'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
#TOP 5 occuring Resource Unit Price
temp = resources['Resource Unit Price'].value_counts().head()
temp.plot.bar()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values')
# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(donations.sample(1000).drop(["Project ID","Donation ID","Donor ID","Donation Received Date"], axis=1), "Donation Included Optional Donation")