# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import glob,os;
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt; # plotting and visualization of the data
import seaborn as sns;

# Input data files are available in the "../input/" directory.
#We first read all the articles info

with open("../input/ArticlesJan2017.csv","rb") as jan17:
    aj17 = pd.read_csv(jan17);
    ArticleJan17 = pd.DataFrame(aj17);
with open("../input/ArticlesFeb2017.csv","rb") as feb17:
    af17 = pd.read_csv(feb17);
    ArticleFeb17 = pd.DataFrame(af17);
with open("../input/ArticlesMarch2017.csv","rb") as mar17:
    am17 = pd.read_csv(mar17);
    ArticleMar17 = pd.DataFrame(am17);   
with open("../input/ArticlesApril2017.csv","rb") as apr17:
    aa17 = pd.read_csv(apr17);
    ArticleApr17 = pd.DataFrame(aa17);
with open("../input/ArticlesMay2017.csv","rb") as may17:
    ay17 = pd.read_csv(may17);
    ArticleMay17 = pd.DataFrame(ay17);
with open("../input/ArticlesJan2018.csv","rb") as jan18:
    aj18 = pd.read_csv(jan18);
    ArticleJan18 = pd.DataFrame(aj18);
with open("../input/ArticlesFeb2018.csv","rb") as feb18:
    af18 = pd.read_csv(feb18);
    ArticleFeb18 = pd.DataFrame(af18);
with open("../input/ArticlesMarch2018.csv","rb") as mar18:
    am18 = pd.read_csv(mar18);
    ArticleMar18 = pd.DataFrame(am18);   
with open("../input/ArticlesApril2018.csv","rb") as apr18:
    aa18 = pd.read_csv(apr18);
    ArticleApr18 = pd.DataFrame(aa18);

numOfArticles = [ArticleJan17['articleID'].count(), ArticleFeb17['articleID'].count(), ArticleMar17['articleID'].count(), ArticleApr17['articleID'].count(), ArticleMay17['articleID'].count(),ArticleJan18['articleID'].count(), ArticleFeb18['articleID'].count(), ArticleMar18['articleID'].count(), ArticleApr18['articleID'].count()];

df = pd.concat([ArticleJan17,ArticleFeb17,ArticleMar17,ArticleApr17,ArticleMay17,ArticleJan18,ArticleFeb18,ArticleMar18,ArticleApr18], sort = False);
print('Total number of articles published in 2017(Jan to May) and 2018(Jan to April):\n', df['articleID'].count());


###Graph 1;The new desk
ndesk = df.groupby('newDesk').size();
plt.figure(1,figsize = ([16,16]));
ndesk.sort_values(ascending = False).plot.bar(color = 'k',alpha = 0.75);
plt.title('CATEGORY OF THE ARTICLE');
plt.ylabel('# of article');
plt.show();

section = df.groupby('sectionName').size();
unknown_to_drop = section.nlargest(1);#find 'Unknown'
section = section.drop(index = unknown_to_drop.index); #drop it
sections_two_categories = unknown_to_drop.append(pd.DataFrame([section.sum()], index = ["other"]));

plt.figure(2,figsize = ([12,12]));
plt.pie(x = sections_two_categories,labels = sections_two_categories.index,autopct = '%1.1f%%');
plt.title('Section where articles published')
plt.show();

plt.figure(3,figsize=([16,16]));
section.sort_values(ascending = False).plot.bar(color='k',alpha=0.75);
plt.ylabel('# of articles');
plt.title('Known section categorisation');
plt.show();

material = df.groupby('typeOfMaterial').size();
plt.figure(4,figsize=([16,16]));
material.sort_values().plot.barh(color = 'k',alpha=0.75);
plt.xlabel('# of articles');
plt.show();

dtype = df.groupby('documentType').size();
plt.figure(5,figsize = ([12,12]));
plt.pie(x = dtype,labels = dtype.index);
plt.show();
page = df.groupby('printPage').size();
plt.figure(6,figsize = ([16,16]));
page.plot(xlim = [0,60]);
plt.ylabel('# of articles');
plt.show();
author = df.groupby('byline').size();
author = author.nlargest(25);
plt.figure(7,figsize = ([16,16]));
author.sort_values().plot.barh(color = 'k', alpha  = 0.7);
plt.xlabel('# of articles');
plt.show();