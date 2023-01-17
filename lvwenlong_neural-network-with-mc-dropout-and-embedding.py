# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy  as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from torch.utils.data import TensorDataset, DataLoader

class Preproc:

    def __init__(self):

        self.train, self.valid = train_test_split(pd.read_csv('../input/train.csv'), test_size = 0.1)

        self.test              = pd.read_csv('../input/test.csv')

        self.fix() # fix wrong data according to discussion

        self.proc_collection()

        self.proc_budget()

        self.proc_genres()

        self.proc_popularity()

        self.proc_runtime()

        self.proc_homepage()

        self.proc_overview()

        self.proc_companies()

        self.proc_countries()

        self.proc_spoken_languages()

        self.proc_Keywords()

        self.proc_cast()

        self.proc_crew()

        self.proc_release_date()

        self.proc_status()

        self.label_encoding()

        self.proc_drop()

        self.train.revenue = self.train.revenue.apply(np.log)

        self.valid.revenue = self.valid.revenue.apply(np.log)

    

    def fix(self):

        self.train.loc[self.train['id'] == 16,'revenue']   = 192864

        self.train.loc[self.train['id'] == 90,'budget']    = 30000000

        self.train.loc[self.train['id'] == 118,'budget']   = 60000000

        self.train.loc[self.train['id'] == 149,'budget']   = 18000000

        self.train.loc[self.train['id'] == 313,'revenue']  = 12000000

        self.train.loc[self.train['id'] == 451,'revenue']  = 12000000

        self.train.loc[self.train['id'] == 464,'budget']   = 20000000

        self.train.loc[self.train['id'] == 470,'budget']   = 13000000

        self.train.loc[self.train['id'] == 513,'budget']   = 930000

        self.train.loc[self.train['id'] == 797,'budget']   = 8000000

        self.train.loc[self.train['id'] == 819,'budget']   = 90000000

        self.train.loc[self.train['id'] == 850,'budget']   = 90000000

        self.train.loc[self.train['id'] == 1007,'budget']  = 2

        self.train.loc[self.train['id'] == 1112,'budget']  = 7500000

        self.train.loc[self.train['id'] == 1131,'budget']  = 4300000

        self.train.loc[self.train['id'] == 1359,'budget']  = 10000000

        self.train.loc[self.train['id'] == 1542,'budget']  = 1

        self.train.loc[self.train['id'] == 1570,'budget']  = 15800000

        self.train.loc[self.train['id'] == 1571,'budget']  = 4000000

        self.train.loc[self.train['id'] == 1714,'budget']  = 46000000

        self.train.loc[self.train['id'] == 1721,'budget']  = 17500000

        self.train.loc[self.train['id'] == 1865,'revenue'] = 25000000

        self.train.loc[self.train['id'] == 1885,'budget']  = 12

        self.train.loc[self.train['id'] == 2091,'budget']  = 10

        self.train.loc[self.train['id'] == 2268,'budget']  = 17500000

        self.train.loc[self.train['id'] == 2491,'budget']  = 6

        self.train.loc[self.train['id'] == 2602,'budget']  = 31000000

        self.train.loc[self.train['id'] == 2612,'budget']  = 15000000

        self.train.loc[self.train['id'] == 2696,'budget']  = 10000000

        self.train.loc[self.train['id'] == 2801,'budget']  = 10000000

        self.train.loc[self.train['id'] == 335,'budget']   = 2

        self.train.loc[self.train['id'] == 348,'budget']   = 12

        self.train.loc[self.train['id'] == 470,'budget']   = 13000000

        self.train.loc[self.train['id'] == 513,'budget']   = 1100000

        self.train.loc[self.train['id'] == 640,'budget']   = 6

        self.train.loc[self.train['id'] == 696,'budget']   = 1

        self.train.loc[self.train['id'] == 797,'budget']   = 8000000

        self.train.loc[self.train['id'] == 850,'budget']   = 1500000

        self.train.loc[self.train['id'] == 1199,'budget']  = 5

        self.train.loc[self.train['id'] == 1282,'budget']  = 9

        self.train.loc[self.train['id'] == 1347,'budget']  = 1

        self.train.loc[self.train['id'] == 1755,'budget']  = 2

        self.train.loc[self.train['id'] == 1801,'budget']  = 5

        self.train.loc[self.train['id'] == 1918,'budget']  = 592

        self.train.loc[self.train['id'] == 2033,'budget']  = 4

        self.train.loc[self.train['id'] == 2118,'budget']  = 344

        self.train.loc[self.train['id'] == 2252,'budget']  = 130

        self.train.loc[self.train['id'] == 2256,'budget']  = 1

        self.train.loc[self.train['id'] == 2696,'budget']  = 10000000





        self.valid.loc[self.valid['id'] == 16,'revenue']   = 192864

        self.valid.loc[self.valid['id'] == 90,'budget']    = 30000000

        self.valid.loc[self.valid['id'] == 118,'budget']   = 60000000

        self.valid.loc[self.valid['id'] == 149,'budget']   = 18000000

        self.valid.loc[self.valid['id'] == 313,'revenue']  = 12000000

        self.valid.loc[self.valid['id'] == 451,'revenue']  = 12000000

        self.valid.loc[self.valid['id'] == 464,'budget']   = 20000000

        self.valid.loc[self.valid['id'] == 470,'budget']   = 13000000

        self.valid.loc[self.valid['id'] == 513,'budget']   = 930000

        self.valid.loc[self.valid['id'] == 797,'budget']   = 8000000

        self.valid.loc[self.valid['id'] == 819,'budget']   = 90000000

        self.valid.loc[self.valid['id'] == 850,'budget']   = 90000000

        self.valid.loc[self.valid['id'] == 1007,'budget']  = 2

        self.valid.loc[self.valid['id'] == 1112,'budget']  = 7500000

        self.valid.loc[self.valid['id'] == 1131,'budget']  = 4300000

        self.valid.loc[self.valid['id'] == 1359,'budget']  = 10000000

        self.valid.loc[self.valid['id'] == 1542,'budget']  = 1

        self.valid.loc[self.valid['id'] == 1570,'budget']  = 15800000

        self.valid.loc[self.valid['id'] == 1571,'budget']  = 4000000

        self.valid.loc[self.valid['id'] == 1714,'budget']  = 46000000

        self.valid.loc[self.valid['id'] == 1721,'budget']  = 17500000

        self.valid.loc[self.valid['id'] == 1865,'revenue'] = 25000000

        self.valid.loc[self.valid['id'] == 1885,'budget']  = 12

        self.valid.loc[self.valid['id'] == 2091,'budget']  = 10

        self.valid.loc[self.valid['id'] == 2268,'budget']  = 17500000

        self.valid.loc[self.valid['id'] == 2491,'budget']  = 6

        self.valid.loc[self.valid['id'] == 2602,'budget']  = 31000000

        self.valid.loc[self.valid['id'] == 2612,'budget']  = 15000000

        self.valid.loc[self.valid['id'] == 2696,'budget']  = 10000000

        self.valid.loc[self.valid['id'] == 2801,'budget']  = 10000000

        self.valid.loc[self.valid['id'] == 335,'budget']   = 2

        self.valid.loc[self.valid['id'] == 348,'budget']   = 12

        self.valid.loc[self.valid['id'] == 470,'budget']   = 13000000

        self.valid.loc[self.valid['id'] == 513,'budget']   = 1100000

        self.valid.loc[self.valid['id'] == 640,'budget']   = 6

        self.valid.loc[self.valid['id'] == 696,'budget']   = 1

        self.valid.loc[self.valid['id'] == 797,'budget']   = 8000000

        self.valid.loc[self.valid['id'] == 850,'budget']   = 1500000

        self.valid.loc[self.valid['id'] == 1199,'budget']  = 5

        self.valid.loc[self.valid['id'] == 1282,'budget']  = 9

        self.valid.loc[self.valid['id'] == 1347,'budget']  = 1

        self.valid.loc[self.valid['id'] == 1755,'budget']  = 2

        self.valid.loc[self.valid['id'] == 1801,'budget']  = 5

        self.valid.loc[self.valid['id'] == 1918,'budget']  = 592

        self.valid.loc[self.valid['id'] == 2033,'budget']  = 4

        self.valid.loc[self.valid['id'] == 2118,'budget']  = 344

        self.valid.loc[self.valid['id'] == 2252,'budget']  = 130

        self.valid.loc[self.valid['id'] == 2256,'budget']  = 1

        self.valid.loc[self.valid['id'] == 2696,'budget']  = 10000000



        self.test.loc[self.test['id'] == 3033,'budget'] = 250 

        self.test.loc[self.test['id'] == 3051,'budget'] = 50

        self.test.loc[self.test['id'] == 3084,'budget'] = 337

        self.test.loc[self.test['id'] == 3224,'budget'] = 4  

        self.test.loc[self.test['id'] == 3594,'budget'] = 25  

        self.test.loc[self.test['id'] == 3619,'budget'] = 500  

        self.test.loc[self.test['id'] == 3831,'budget'] = 3  

        self.test.loc[self.test['id'] == 3935,'budget'] = 500  

        self.test.loc[self.test['id'] == 4049,'budget'] = 995946 

        self.test.loc[self.test['id'] == 4424,'budget'] = 3  

        self.test.loc[self.test['id'] == 4460,'budget'] = 8  

        self.test.loc[self.test['id'] == 4555,'budget'] = 1200000 

        self.test.loc[self.test['id'] == 4624,'budget'] = 30 

        self.test.loc[self.test['id'] == 4645,'budget'] = 500 

        self.test.loc[self.test['id'] == 4709,'budget'] = 450 

        self.test.loc[self.test['id'] == 4839,'budget'] = 7

        self.test.loc[self.test['id'] == 3125,'budget'] = 25 

        self.test.loc[self.test['id'] == 3142,'budget'] = 1

        self.test.loc[self.test['id'] == 3201,'budget'] = 450

        self.test.loc[self.test['id'] == 3222,'budget'] = 6

        self.test.loc[self.test['id'] == 3545,'budget'] = 38

        self.test.loc[self.test['id'] == 3670,'budget'] = 18

        self.test.loc[self.test['id'] == 3792,'budget'] = 19

        self.test.loc[self.test['id'] == 3881,'budget'] = 7

        self.test.loc[self.test['id'] == 3969,'budget'] = 400

        self.test.loc[self.test['id'] == 4196,'budget'] = 6

        self.test.loc[self.test['id'] == 4221,'budget'] = 11

        self.test.loc[self.test['id'] == 4222,'budget'] = 500

        self.test.loc[self.test['id'] == 4285,'budget'] = 11

        self.test.loc[self.test['id'] == 4319,'budget'] = 1

        self.test.loc[self.test['id'] == 4639,'budget'] = 10

        self.test.loc[self.test['id'] == 4719,'budget'] = 45

        self.test.loc[self.test['id'] == 4822,'budget'] = 22

        self.test.loc[self.test['id'] == 4829,'budget'] = 20

        self.test.loc[self.test['id'] == 4969,'budget'] = 20

        self.test.loc[self.test['id'] == 5021,'budget'] = 40 

        self.test.loc[self.test['id'] == 5035,'budget'] = 1 

        self.test.loc[self.test['id'] == 5063,'budget'] = 14 

        self.test.loc[self.test['id'] == 5119,'budget'] = 2 

        self.test.loc[self.test['id'] == 5214,'budget'] = 30 

        self.test.loc[self.test['id'] == 5221,'budget'] = 50 

        self.test.loc[self.test['id'] == 4903,'budget'] = 15

        self.test.loc[self.test['id'] == 4983,'budget'] = 3

        self.test.loc[self.test['id'] == 5102,'budget'] = 28

        self.test.loc[self.test['id'] == 5217,'budget'] = 75

        self.test.loc[self.test['id'] == 5224,'budget'] = 3 

        self.test.loc[self.test['id'] == 5469,'budget'] = 20 

        self.test.loc[self.test['id'] == 5840,'budget'] = 1 

        self.test.loc[self.test['id'] == 5960,'budget'] = 30

        self.test.loc[self.test['id'] == 6506,'budget'] = 11 

        self.test.loc[self.test['id'] == 6553,'budget'] = 280

        self.test.loc[self.test['id'] == 6561,'budget'] = 7

        self.test.loc[self.test['id'] == 6582,'budget'] = 218

        self.test.loc[self.test['id'] == 6638,'budget'] = 5

        self.test.loc[self.test['id'] == 6749,'budget'] = 8 

        self.test.loc[self.test['id'] == 6759,'budget'] = 50 

        self.test.loc[self.test['id'] == 6856,'budget'] = 10

        self.test.loc[self.test['id'] == 6858,'budget'] =  100

        self.test.loc[self.test['id'] == 6876,'budget'] =  250

        self.test.loc[self.test['id'] == 6972,'budget'] = 1

        self.test.loc[self.test['id'] == 7079,'budget'] = 8000000

        self.test.loc[self.test['id'] == 7150,'budget'] = 118

        self.test.loc[self.test['id'] == 6506,'budget'] = 118

        self.test.loc[self.test['id'] == 7225,'budget'] = 6

        self.test.loc[self.test['id'] == 7231,'budget'] = 85

        self.test.loc[self.test['id'] == 5222,'budget'] = 5

        self.test.loc[self.test['id'] == 5322,'budget'] = 90

        self.test.loc[self.test['id'] == 5350,'budget'] = 70

        self.test.loc[self.test['id'] == 5378,'budget'] = 10

        self.test.loc[self.test['id'] == 5545,'budget'] = 80

        self.test.loc[self.test['id'] == 5810,'budget'] = 8

        self.test.loc[self.test['id'] == 5926,'budget'] = 300

        self.test.loc[self.test['id'] == 5927,'budget'] = 4

        self.test.loc[self.test['id'] == 5986,'budget'] = 1

        self.test.loc[self.test['id'] == 6053,'budget'] = 20

        self.test.loc[self.test['id'] == 6104,'budget'] = 1

        self.test.loc[self.test['id'] == 6130,'budget'] = 30

        self.test.loc[self.test['id'] == 6301,'budget'] = 150

        self.test.loc[self.test['id'] == 6276,'budget'] = 100

        self.test.loc[self.test['id'] == 6473,'budget'] = 100

        self.test.loc[self.test['id'] == 6842,'budget'] = 30

    

    def proc_collection(self):

        self.train.belongs_to_collection = self.train.belongs_to_collection.notna()

        self.valid.belongs_to_collection = self.valid.belongs_to_collection.notna()

        self.test.belongs_to_collection  = self.test.belongs_to_collection.notna()

    

    def proc_budget(self):

        self.train['median_budget'] = self.train.budget == 0

        self.valid['median_budget'] = self.valid.budget == 0

        self.test['median_budget']  = self.test.budget  == 0

        median_budget = self.train.loc[self.train.median_budget == False, 'budget'].median()



        self.train.loc[self.train.median_budget, 'budget'] = median_budget

        self.valid.loc[self.valid.median_budget, 'budget'] = median_budget

        self.test.loc[self.test.median_budget, 'budget']   = median_budget



        self.train.budget = self.train.budget.apply(np.log)

        self.valid.budget = self.valid.budget.apply(np.log)

        self.test.budget  = self.test.budget.apply(np.log)

        

        mu  = self.train.budget.mean()

        std = self.train.budget.std()



        self.train.budget = self.train.budget.apply(lambda x : (x - mu) / std)

        self.valid.budget = self.valid.budget.apply(lambda x : (x - mu) / std)

        self.test.budget  = self.test.budget.apply(lambda x : (x - mu) / std)



    def proc_popularity(self):

        self.train.popularity = self.train.popularity.apply(np.sqrt)

        self.valid.popularity = self.valid.popularity.apply(np.sqrt)

        self.test.popularity  = self.test.popularity.apply(np.sqrt)

        

        mu  = self.train.popularity.mean()

        std = self.train.popularity.std()



        self.train.popularity = self.train.popularity.apply(lambda x : (x - mu) / std)

        self.valid.popularity = self.valid.popularity.apply(lambda x : (x - mu) / std)

        self.test.popularity  = self.test.popularity.apply(lambda x : (x - mu) / std)



    def proc_runtime(self):

        median_runtime = self.train.runtime.median()

        self.train.runtime.fillna(median_runtime, inplace = True)

        self.valid.runtime.fillna(median_runtime, inplace = True)

        self.test.runtime.fillna(median_runtime, inplace = True)



        mu  = self.train.runtime.mean()

        std = self.train.runtime.std()



        self.train.runtime = self.train.runtime.apply(lambda x : (x - mu) / std)

        self.valid.runtime = self.valid.runtime.apply(lambda x : (x - mu) / std)

        self.test.runtime  = self.test.runtime.apply(lambda x : (x - mu) / std)



    def proc_genres(self):

        self.train.genres.fillna('[]', inplace =  True)

        self.valid.genres.fillna('[]', inplace =  True)

        self.test.genres.fillna('[]', inplace =  True)



        genres  = pd.concat([self.train.genres, self.valid.genres, self.test.genres])

        gen_set = set()

        for rec in genres:

            for g in eval(rec):

                gen_set.add(g['name'])

        for g in gen_set:

            self.train[g] = self.train.genres.apply(lambda rec : g in rec)

            self.valid[g] = self.valid.genres.apply(lambda rec : g in rec)

            self.test[g]  = self.test.genres.apply(lambda rec : g in rec)

        self.train.drop(columns = 'genres', inplace = True)

        self.valid.drop(columns = 'genres', inplace = True)

        self.test.drop( columns = 'genres', inplace = True)

    

    def proc_homepage(self):

        self.train.homepage = self.train.homepage.notna()

        self.valid.homepage = self.valid.homepage.notna()

        self.test.homepage  = self.test.homepage.notna()

    

    def proc_imdb(self):

        self.train.drop(columns = 'imdb_id', inplace = True)

        self.valid.drop(columns = 'imdb_id', inplace = True)

        self.test.drop(columns = 'imdb_id', inplace = True)



    def proc_drop(self):

        self.train.drop(columns = ['imdb_id', 'original_title', 'poster_path', 'title', 'tagline'], inplace = True)

        self.valid.drop(columns = ['imdb_id', 'original_title', 'poster_path', 'title', 'tagline'], inplace = True)

        self.test.drop(columns  = ['imdb_id', 'original_title', 'poster_path', 'title', 'tagline'], inplace = True)

    

    def proc_overview(self):

        self.train.overview = self.train.overview.fillna('').apply(lambda rec : len(rec.split()))

        self.valid.overview = self.valid.overview.fillna('').apply(lambda rec : len(rec.split()))

        self.test.overview  = self.test.overview.fillna('').apply(lambda rec : len(rec.split()))



    def proc_companies(self):

        self.train.production_companies = self.train.production_companies.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.production_companies = self.valid.production_companies.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.production_companies  = self.test.production_companies.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_countries(self):

        self.train.production_countries = self.train.production_countries.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.production_countries = self.valid.production_countries.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.production_countries  = self.test.production_countries.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_spoken_languages(self):

        self.train.spoken_languages = self.train.spoken_languages.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.spoken_languages = self.valid.spoken_languages.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.spoken_languages  = self.test.spoken_languages.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_Keywords(self):

        self.train.Keywords = self.train.Keywords.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.Keywords = self.valid.Keywords.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.Keywords  = self.test.Keywords.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_cast(self):

        self.train.cast = self.train.cast.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.cast = self.valid.cast.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.cast  = self.test.cast.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_crew(self):

        self.train.crew = self.train.crew.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.valid.crew = self.valid.crew.fillna('[]').apply(lambda rec : len(eval(rec)))

        self.test.crew  = self.test.crew.fillna('[]').apply(lambda rec : len(eval(rec)))



    def proc_release_date(self):

        self.train['release_year'] = self.train.release_date.astype('datetime64').apply(lambda rec : rec.year if rec.year < 2019 else rec.year - 100)

        self.valid['release_year'] = self.valid.release_date.astype('datetime64').apply(lambda rec : rec.year if rec.year < 2019 else rec.year - 100)

        self.test['release_year']  = self.test.release_date.astype('datetime64').apply(lambda rec : rec.year  if rec.year < 2019 else rec.year - 100)

        self.test.release_year.fillna(self.train.release_year.median(), inplace = True)



        self.train['release_month'] = self.train.release_date.astype('datetime64').apply(lambda rec : rec.month)

        self.valid['release_month'] = self.valid.release_date.astype('datetime64').apply(lambda rec : rec.month)

        self.test['release_month']  = self.test.release_date.astype('datetime64').apply(lambda rec  : rec.month)

        self.test.release_month.fillna(self.train.release_month.median(), inplace = True)



        self.train.drop(columns = 'release_date', inplace = True)

        self.valid.drop(columns = 'release_date', inplace = True)

        self.test.drop( columns = 'release_date', inplace = True)

    

    def proc_status(self):

        self.test.status.fillna('Released', inplace = True)

    

    def label_encoding(self):

        self.cat_names  = [ 'belongs_to_collection' , 'homepage' , 'original_language' , 'status' , 'median_budget' , 'TV Movie' , 'Drama' , 'Music' , 'Horror' , 'Foreign' , 'War' , 'Romance' , 'Mystery' , 'Western' , 'Thriller' , 'Science Fiction' , 'Action' , 'Adventure' , 'Animation' , 'Fantasy' , 'History' , 'Documentary' , 'Comedy' , 'Family' , 'Crime' , 'release_month']

        self.target     = 'revenue'

        

        self.cat_nums = []

        for cat in self.cat_names:

            all_cat         = pd.concat([self.train[cat], self.valid[cat], self.test[cat]])

            encoder         = LabelEncoder().fit(all_cat)

            self.train[cat] = encoder.transform(self.train[cat])

            self.valid[cat] = encoder.transform(self.valid[cat])

            self.test[cat]  = encoder.transform(self.test[cat])

            self.cat_nums.append(len(encoder.classes_))

    

    def cont_normalization(self):

        self.cont_names = [ 'budget' , 'overview' , 'popularity' , 'production_companies' , 'production_countries' , 'runtime' , 'spoken_languages' , 'Keywords' , 'cast' , 'crew' , 'release_year']

        for cont in self.cont_names:

            mean             = self.train[cont].mean()

            std              = self.train[cont].std()

            self.train[cont] = self.train[cont].apply(lambda x : (x - mean) / std)

            self.valid[cont] = self.valid[cont].apply(lambda x : (x - mean) / std)

            self.test[cont]  = self.test[cont].apply(lambda x  : (x - mean) / std)

proc = Preproc()

proc.train.head()
proc.valid.head()
proc.test.head()
import numpy  as np

import torch

import torch.nn as nn



class EmbModel(nn.Module):

    def __init__(self, cat_nums, dim, num_hiddens = 50, dr = 0.01):

        super(EmbModel, self).__init__()

        self.cat_nums       = cat_nums

        self.total_emb_size = 0

        self.emb_list       = nn.ModuleList()

        for cn in cat_nums:

            emb_size = min(50, (cn + 1) // 2)

            self.emb_list.append(nn.Embedding(cn, emb_size))

            self.total_emb_size += emb_size

        

        self.emb_drop = nn.Dropout(p = dr)

        self.mlp = nn.Sequential(

                nn.Linear(self.total_emb_size + dim, num_hiddens), 

                nn.ReLU(inplace = True), 

                nn.Dropout(p = dr), 

                nn.Linear(num_hiddens, num_hiddens), 

                nn.ReLU(inplace = True), 

                nn.Dropout(p = dr), 

                nn.Linear(num_hiddens, 1)

        )

    def forward(self, xcat, xcont):

        embs   = torch.cat([self.emb_list[i](xcat[:, i]) for i in range(len(self.cat_nums))], dim = 1)

        mlp_in = torch.cat((self.emb_drop(embs), xcont), dim = 1)

        return self.mlp(mlp_in)

proc.cont_normalization()

xcat = torch.LongTensor(proc.train[proc.cat_names].values.astype(np.int64))

xcont = torch.FloatTensor(proc.train[proc.cont_names].values)

y = torch.FloatTensor(proc.train.revenue.values)



vxcat = torch.LongTensor(proc.valid[proc.cat_names].values.astype(np.int64))

vxcont = torch.FloatTensor(proc.valid[proc.cont_names].values)

vy = torch.FloatTensor(proc.valid.revenue.values)



nn = EmbModel(proc.cat_nums, len(proc.cont_names),num_hiddens=100, dr = 0.3)

opt = torch.optim.Adam(nn.parameters(), lr = 3e-3)

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda iter: (1+iter)**-0.3)

crit = torch.nn.MSELoss(reduction='sum')





ym = y.mean()

ystd = y.std()



y = (y - ym) / ystd

vy = (vy - ym) / ystd



ds = TensorDataset(xcat,xcont,y)

dl = DataLoader(ds,shuffle=True,batch_size=32)
for epoch in range(60):

    mse = 0.

    for bx_cat,bx_cont,by in dl:

        py = nn(bx_cat,bx_cont).squeeze()

        loss = crit(py,by)

        mse += loss.detach().item()

        opt.zero_grad()

        loss.backward()

        opt.step()

        scheduler.step()

    

    vmse = crit(vy, nn(vxcat,vxcont).squeeze()) / vy.numel()

    print('Epoch %4d loss = %g %g' % (epoch, ystd**2 * mse / y.numel(), ystd**2 * vmse))


tx_cat = torch.LongTensor(proc.test[proc.cat_names].values.astype(np.int64))

tx_cont = torch.FloatTensor(proc.test[proc.cont_names].values.astype(np.float64))



num_ens = 1000

pred = torch.zeros(num_ens, tx_cat.shape[0])

with torch.no_grad():

    for i in range(num_ens):

        pred[i] = ystd * nn(tx_cat, tx_cont).squeeze() + ym



pred_test = pred.mean(dim = 0).exp().numpy()
proc.test.set_index('id', inplace = True)

proc.test['revenue'] = pred_test
proc.test.loc[:,['revenue']].to_csv('submission.csv')