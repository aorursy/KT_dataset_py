import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from treeviz_util import tree_print
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

df.head()
df_shrooms = pd.DataFrame()



# class: edible=e, poisonous=p

df_shrooms['edible'] = df['class'] == 'e'



# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

df_shrooms['cap-shape-bell'] = df['cap-shape'] == 'b'

df_shrooms['cap-shape-conical'] = df['cap-shape'] == 'c'

df_shrooms['cap-shape-convex'] = df['cap-shape'] == 'x'

df_shrooms['cap-shape-flat'] = df['cap-shape'] == 'f'

df_shrooms['cap-shape-knobbed'] = df['cap-shape'] == 'k'

df_shrooms['cap-shape-sunken'] = df['cap-shape'] == 's'



# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

df_shrooms['cap-surface-fibrous'] = df['cap-surface'] == 'f'

df_shrooms['cap-surface-grooves'] = df['cap-surface'] == 'g'

df_shrooms['cap-surface-scaly'] = df['cap-surface'] == 'y'

df_shrooms['cap-surface-smooth'] = df['cap-surface'] == 's'



# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

df_shrooms['cap-color-brown'] = df['cap-color'] == 'n'

df_shrooms['cap-color-buff'] = df['cap-color'] == 'b'

df_shrooms['cap-color-cinnamon'] = df['cap-color'] == 'c'

df_shrooms['cap-color-gray'] = df['cap-color'] == 'g'

df_shrooms['cap-color-green'] = df['cap-color'] == 'r'

df_shrooms['cap-color-pink'] = df['cap-color'] == 'p'

df_shrooms['cap-color-purple'] = df['cap-color'] == 'u'

df_shrooms['cap-color-red'] = df['cap-color'] == 'e'

df_shrooms['cap-color-white'] = df['cap-color'] == 'w'

df_shrooms['cap-color-yellow'] = df['cap-color'] == 'y'



# bruises: bruises=t,no=f

df_shrooms['bruises'] = df['bruises'] == 't'



# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

df_shrooms['odor-almond'] = df['odor'] == 'a'

df_shrooms['odor-anise'] = df['odor'] == 'l'

df_shrooms['odor-creosote'] = df['odor'] == 'c'

df_shrooms['odor-fishy'] = df['odor'] == 'y'

df_shrooms['odor-foul'] = df['odor'] == 'f'

df_shrooms['odor-musty'] = df['odor'] == 'm'

df_shrooms['odor-none'] = df['odor'] == 'n'

df_shrooms['odor-pungent'] = df['odor'] == 'p'

df_shrooms['odor-spicy'] = df['odor'] == 's'



# gill-attachment: attached=a,descending=d,free=f,notched=n

df_shrooms['gill-attachment-attached'] = df['gill-attachment'] == 'a'

df_shrooms['gill-attachment-descending'] = df['gill-attachment'] == 'd'

df_shrooms['gill-attachment-free'] = df['gill-attachment'] == 'f'

df_shrooms['gill-attachment-notched'] = df['gill-attachment'] == 'n'



# gill-spacing: close=c,crowded=w,distant=d

df_shrooms['gill-spacing-close'] = df['gill-spacing'] == 'c'

df_shrooms['gill-spacing-crowded'] = df['gill-spacing'] == 'w'

df_shrooms['gill-spacing-distant'] = df['gill-spacing'] == 'd'



# gill-size: broad=b,narrow=n

df_shrooms['broad-gill-size'] = df['gill-size'] == 'b'



# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

df_shrooms['broad-color-black'] = df['gill-color'] == 'k'

df_shrooms['broad-color-brown'] = df['gill-color'] == 'n'

df_shrooms['broad-color-buff'] = df['gill-color'] == 'b'

df_shrooms['broad-color-chocolate'] = df['gill-color'] == 'h'

df_shrooms['broad-color-gray'] = df['gill-color'] == 'g'

df_shrooms['broad-color-green'] = df['gill-color'] == 'r'

df_shrooms['broad-color-orange'] = df['gill-color'] == 'o'

df_shrooms['broad-color-pink'] = df['gill-color'] == 'p'

df_shrooms['broad-color-purple'] = df['gill-color'] == 'u'

df_shrooms['broad-color-red'] = df['gill-color'] == 'e'

df_shrooms['broad-color-white'] = df['gill-color'] == 'w'

df_shrooms['broad-color-yellow'] = df['gill-color'] == 'y'



# stalk-shape: enlarging=e,tapering=t

df_shrooms['enlarging-stalk-shape'] = df['stalk-shape'] == 'e'



# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

df_shrooms['stalk-root-bulbous'] = df['stalk-root'] == 'b'

df_shrooms['stalk-root-club'] = df['stalk-root'] == 'c'

df_shrooms['stalk-root-cup'] = df['stalk-root'] == 'u'

df_shrooms['stalk-root-equal'] = df['stalk-root'] == 'e'

df_shrooms['stalk-root-rhizomorphs'] = df['stalk-root'] == 'z'

df_shrooms['stalk-root-rooted'] = df['stalk-root'] == 'r'

df_shrooms['stalk-root-missing'] = df['stalk-root'] == '?'



# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

df_shrooms['stalk-surface-above-ring-fibrous'] = df['stalk-surface-above-ring'] == 'f'

df_shrooms['stalk-surface-above-ring-scaly'] = df['stalk-surface-above-ring'] == 'y'

df_shrooms['stalk-surface-above-ring-silky'] = df['stalk-surface-above-ring'] == 'k'

df_shrooms['stalk-surface-above-ring-smooth'] = df['stalk-surface-above-ring'] == 's'



# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

df_shrooms['stalk-surface-below-ring-fibrous'] = df['stalk-surface-below-ring'] == 'f'

df_shrooms['stalk-surface-below-ring-scaly'] = df['stalk-surface-below-ring'] == 'y'

df_shrooms['stalk-surface-below-ring-silky'] = df['stalk-surface-below-ring'] == 'k'

df_shrooms['stalk-surface-below-ring-smooth'] = df['stalk-surface-below-ring'] == 's'



# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

df_shrooms['stalk-color-above-ring-brown'] = df['stalk-color-above-ring'] == 'n'

df_shrooms['stalk-color-above-ring-buff'] = df['stalk-color-above-ring'] == 'b'

df_shrooms['stalk-color-above-ring-cinnamon'] = df['stalk-color-above-ring'] == 'c'

df_shrooms['stalk-color-above-ring-gray'] = df['stalk-color-above-ring'] == 'g'

df_shrooms['stalk-color-above-ring-orange'] = df['stalk-color-above-ring'] == 'o'

df_shrooms['stalk-color-above-ring-pink'] = df['stalk-color-above-ring'] == 'p'

df_shrooms['stalk-color-above-ring-red'] = df['stalk-color-above-ring'] == 'e'

df_shrooms['stalk-color-above-ring-white'] = df['stalk-color-above-ring'] == 'w'

df_shrooms['stalk-color-above-ring-yellow'] = df['stalk-color-above-ring'] == 'y'



# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

df_shrooms['stalk-color-below-ring-brown'] = df['stalk-color-below-ring'] == 'n'

df_shrooms['stalk-color-below-ring-buff'] = df['stalk-color-below-ring'] == 'b'

df_shrooms['stalk-color-below-ring-cinnamon'] = df['stalk-color-below-ring'] == 'c'

df_shrooms['stalk-color-below-ring-gray'] = df['stalk-color-below-ring'] == 'g'

df_shrooms['stalk-color-below-ring-orange'] = df['stalk-color-below-ring'] == 'o'

df_shrooms['stalk-color-below-ring-pink'] = df['stalk-color-below-ring'] == 'p'

df_shrooms['stalk-color-below-ring-red'] = df['stalk-color-below-ring'] == 'e'

df_shrooms['stalk-color-below-ring-white'] = df['stalk-color-below-ring'] == 'w'

df_shrooms['stalk-color-below-ring-yellow'] = df['stalk-color-below-ring'] == 'y'



# veil-type: partial=p,universal=u

df_shrooms['partial-veil-type'] = df['veil-type'] == 'p'



# veil-color: brown=n,orange=o,white=w,yellow=y

df_shrooms['veil-color-brown'] = df['veil-color'] == 'n'

df_shrooms['veil-color-orange'] = df['veil-color'] == 'o'

df_shrooms['veil-color-white'] = df['veil-color'] == 'w'

df_shrooms['veil-color-yellow'] = df['veil-color'] == 'y'



# ring-number: none=n,one=o,two=t

df_shrooms['ring-number-none'] = df['ring-number'] == 'n'

df_shrooms['ring-number-one'] = df['ring-number'] == 'o'

df_shrooms['ring-number-two'] = df['ring-number'] == 't'



# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

df_shrooms['ring-type-cobwebby'] = df['ring-type'] == 'c'

df_shrooms['ring-type-evanescent'] = df['ring-type'] == 'e'

df_shrooms['ring-type-flaring'] = df['ring-type'] == 'f'

df_shrooms['ring-type-large'] = df['ring-type'] == 'l'

df_shrooms['ring-type-none'] = df['ring-type'] == 'n'

df_shrooms['ring-type-pendant'] = df['ring-type'] == 'p'

df_shrooms['ring-type-sheathing'] = df['ring-type'] == 's'

df_shrooms['ring-type-zone'] = df['ring-type'] == 'z'



# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

df_shrooms['spore-print-color-black'] = df['spore-print-color'] == 'k'

df_shrooms['spore-print-color-brown'] = df['spore-print-color'] == 'n'

df_shrooms['spore-print-color-buff'] = df['spore-print-color'] == 'b'

df_shrooms['spore-print-color-chocolate'] = df['spore-print-color'] == 'h'

df_shrooms['spore-print-color-green'] = df['spore-print-color'] == 'r'

df_shrooms['spore-print-color-orange'] = df['spore-print-color'] == 'o'

df_shrooms['spore-print-color-purple'] = df['spore-print-color'] == 'u'

df_shrooms['spore-print-color-white'] = df['spore-print-color'] == 'w'

df_shrooms['spore-print-color-yellow'] = df['spore-print-color'] == 'y'



# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

df_shrooms['population-abundant'] = df['population'] == 'a'

df_shrooms['population-clustered'] = df['population'] == 'c'

df_shrooms['population-numerous'] = df['population'] == 'n'

df_shrooms['population-scattered'] = df['population'] == 's'

df_shrooms['population-several'] = df['population'] == 'v'

df_shrooms['population-solitary'] = df['population'] == 'y'



# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

df_shrooms['habitat-grasses'] = df['habitat'] == 'g'

df_shrooms['habitat-leaves'] = df['habitat'] == 'l'

df_shrooms['habitat-meadows'] = df['habitat'] == 'm'

df_shrooms['habitat-paths'] = df['habitat'] == 'p'

df_shrooms['habitat-urban'] = df['habitat'] == 'u'

df_shrooms['habitat-waste'] = df['habitat'] == 'w'

df_shrooms['habitat-woods'] = df['habitat'] == 'd'



df_shrooms.head()
dtc = DecisionTreeClassifier(max_depth=5)

cols = list(df_shrooms.columns)

cols.remove('edible')

X=df_shrooms[cols]

y=df_shrooms['edible']

dtc.fit(X, y)
tree_print(dtc, X)
dtc.score(X, y)