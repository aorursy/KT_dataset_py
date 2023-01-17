# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np
ClicksTest=pd.read_csv("../input/clicks_test.csv",encoding="utf8")
ClicksTest.head()
ClicksTest.shape
ClicksTrain=pd.read_csv("../input/clicks_train.csv",encoding="utf8")
ClicksTrain.head()
ClicksTrain.shape
DocumentsCategories=pd.read_csv("../input/documents_categories.csv",encoding="utf8")
DocumentsCategories.head()
DocumentsCategories.shape
DocumentsEntities=pd.read_csv("../input/documents_entities.csv",encoding="utf8")
DocumentsEntities.head()
DocumentsEntities.shape
DocumentsMeta=pd.read_csv("../input/documents_meta.csv",encoding="utf8")
DocumentsMeta.head()
DocumentsMeta.shape
DocumentsTopics=pd.read_csv("../input/documents_topics.csv",encoding="utf8")
DocumentsTopics.head()
DocumentsTopics.shape
Events=pd.read_csv("../input/events.csv",encoding="utf8")
Events.head()
Events.shape
PageViewsSample=pd.read_csv("../input/page_views_sample.csv",encoding="utf8")
PageViewsSample.head()
PageViewsSample.shape
PromotedContent=pd.read_csv("../input/promoted_content.csv",encoding="utf8")
PromotedContent.head()
PromotedContent.shape