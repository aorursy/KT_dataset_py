import pandas as pd
import numpy as np

add = '../input/heroes_information.csv'
hero = pd.read_csv(add)
hero.head()
import matplotlib.pyplot as plt

hero.nunique()
plt.style.available
plt.style.use('fivethirtyeight')
hero.Publisher.value_counts().plot(kind='pie', figsize=(6,6),title='Publisherwise SuperHero Distribution')
gen=hero.Gender.value_counts()
gen
plt.style.use('seaborn')
gen.plot(kind='pie',figsize=(8,8),legend=True)
hero.Alignment.value_counts()
def gender(s):
    if(s=='Male'):
        return 1
    elif(s=='Female'):
        return 0
    else:
        return -1
    
def align(s):
    if(s=='good'):
        return 1
    elif(s=='neutral'):
        return 0
    else:
        return -1
    
hero['sex']= hero.apply(lambda hero:gender(hero['Gender']),axis=1)
hero['align'] = hero.apply(lambda hero:align(hero['Alignment']),axis=1)
hero.head(5)
hero_gender = hero.pivot_table(values='name',index='Publisher',columns=['sex'],aggfunc=np.count_nonzero)
pub = hero.pivot_table(values='name',index='Publisher',aggfunc=np.count_nonzero)
pub.rename(columns={'name':'total'},inplace=True)
pub['male_percent']= hero_gender[1]*100/pub.total
pub_m =pub.sort_values(by=['total'],ascending=False).head(10)
pub_m
pub_m.sort_values('male_percent', ascending = True,inplace=True )
pub_m['male_percent'].plot(kind='barh')
hero_align = hero.pivot_table(values='name',index='align',columns=['sex'],aggfunc=np.count_nonzero)
h_align=hero_align/pub.total.sum()*100
bad_woman_percent = h_align[0][-1]*100/(h_align[0][1]+h_align[0][-1])
bad_woman_percent
bad_men_percent = h_align[1][-1]*100/(h_align[1][1]+h_align[1][-1])
bad_men_percent
bad_men_percent/bad_woman_percent
h_align
h_corr=hero.corr(method='pearson')
print((h_corr>0.5)|(h_corr<-0.5))

hero_color =hero[['Eye color','Hair color','Skin color']]
#you can observe some colors have words starting in capital and some in lower case
#lets convert all of them into lower case
hero_color= hero_color.applymap(lambda x : x.lower())
hero_color['combo']=hero_color['Eye color']+"-"+hero_color['Hair color']+"-"+hero_color['Skin color']
hero_color.combo.value_counts().head(10)
adda="../input/super_hero_powers.csv"
power = pd.read_csv(adda)
power.set_index('hero_names',inplace=True)
power.sum(axis=1).sort_values(ascending=False)[:15]