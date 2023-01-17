import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_sugars = world_food_facts[world_food_facts.sugars_100g.notnull()]

def return_sugars(country):
    return world_sugars[world_sugars.countries == country].sugars_100g.tolist()
    
# Get list of sugars per 100g for some countries
fr_sugars = return_sugars('france') + return_sugars('en:fr')
za_sugars = return_sugars('south africa')
uk_sugars = return_sugars('united kingdom') + return_sugars('en:gb')
us_sugars = return_sugars('united states') + return_sugars('en:us') + return_sugars('us')
sp_sugars = return_sugars('spain') + return_sugars('españa') + return_sugars('en:es')
nd_sugars = return_sugars('netherlands') + return_sugars('holland')
au_sugars = return_sugars('australia') + return_sugars('en:au')
cn_sugars = return_sugars('canada') + return_sugars('en:cn')
de_sugars = return_sugars('germany')
gr_sugars = return_sugars('greece') + return_sugars('greece')

countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'ND', 'AU', 'CN', 'DE', 'GR']
sugars_l = [mean(fr_sugars), 
            mean(za_sugars), 
            mean(uk_sugars), 
            mean(us_sugars), 
            mean(sp_sugars), 
            mean(nd_sugars),
            mean(au_sugars),
            mean(cn_sugars),
            mean(de_sugars),
            mean(gr_sugars)]
            
y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sugars_l, align='center', alpha=0.5)
plt.title('Average total sugar content per 100g')
plt.xticks(y_pos, countries)
plt.ylabel('Sugar/100g')
    
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_sodium = world_food_facts[world_food_facts.sodium_100g.notnull()]

def return_sodium(country):
    return world_sodium[world_sodium.countries == country].sodium_100g.tolist()
    
# Get list of sodium per 100g for some countries
fr_sodium = return_sodium('france') + return_sodium('en:fr')
za_sodium = return_sodium('south africa')
uk_sodium = return_sodium('united kingdom') + return_sodium('en:gb')
us_sodium = return_sodium('united states') + return_sodium('en:us') + return_sodium('us')
sp_sodium = return_sodium('spain') + return_sodium('españa') + return_sodium('en:es')
ch_sodium = return_sodium('china')
nd_sodium = return_sodium('netherlands') + return_sodium('holland')
au_sodium = return_sodium('australia') + return_sodium('en:au')
jp_sodium = return_sodium('japan') + return_sodium('en:jp')
de_sodium = return_sodium('germany')

countries = ['FR', 'ZA', 'UK', 'USA', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']
sodium_l = [mean(fr_sodium), 
            mean(za_sodium), 
            mean(uk_sodium), 
            mean(us_sodium), 
            mean(sp_sodium), 
            mean(ch_sodium),
            mean(nd_sodium),
            mean(au_sodium),
            mean(jp_sodium),
            mean(de_sodium)]

y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sodium_l, align='center', alpha=0.5)
plt.title('Average sodium content per 100g')
plt.xticks(y_pos, countries)
plt.ylabel('Sodium/100g')
    
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

world_food_facts = pd.read_csv('../input/FoodFacts.csv')
world_food_facts.countries = world_food_facts.countries.str.lower()
    
def mean(l):
    return float(sum(l)) / len(l)

world_additives = world_food_facts[world_food_facts.additives_n.notnull()]

def return_additives(country):
    return world_additives[world_additives.countries == country].additives_n.tolist()
    
# Get list of additives amounts for some countries
fr_additives = return_additives('france') + return_additives('en:fr')
za_additives = return_additives('south africa')
uk_additives = return_additives('united kingdom') + return_additives('en:gb')
us_additives = return_additives('united states') + return_additives('en:us') + return_additives('us')
sp_additives = return_additives('spain') + return_additives('españa') + return_additives('en:es')
ch_additives = return_additives('china')
nd_additives = return_additives('netherlands') + return_additives('holland')
au_additives = return_additives('australia') + return_additives('en:au')
jp_additives = return_additives('japan') + return_additives('en:jp')
de_additives = return_additives('germany')

countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'CH', 'ND', 'AU', 'JP', 'DE']
additives_l = [mean(fr_additives), 
            mean(za_additives), 
            mean(uk_additives), 
            mean(us_additives), 
            mean(sp_additives), 
            mean(ch_additives),
            mean(nd_additives),
            mean(au_additives),
            mean(jp_additives),
            mean(de_additives)]

y_pos = np.arange(len(countries))
    
plt.bar(y_pos, sodium_l, align='center', alpha=0.5)
plt.title('Average amount of additives')
plt.xticks(y_pos, countries)
plt.ylabel('Amount of additives')
    
plt.show()
# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.
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