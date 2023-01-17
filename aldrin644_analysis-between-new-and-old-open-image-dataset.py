import pandas as pd
import numpy as np
import json

new_oid = pd.read_csv("../input/data-files/class-descriptions-boxable.csv")
with open('../input/data-files/old_oid_labels.json') as f:
    old_oid = json.load(f)
new_oid_name = list(new_oid['name'])
new_oid_display_name = list(new_oid['display_name'])
new_oid_display_name = [w.lower() for w in new_oid_display_name]
old_oid_name = [w['name'] for w in old_oid]
old_oid_display_name = [w['display_name'].lower() for w in old_oid]
print("New Open Image Dataset ----------->\n")
print(len(new_oid_name),new_oid_name,'\n\n')
print(len(new_oid_display_name),new_oid_display_name,'\n\n')
print("Old Open Image Dataset ----------->\n")
print(len(old_oid_name),old_oid_name,'\n\n')
print(len(old_oid_display_name), old_oid_display_name,'\n\n')
name_intersection = set(new_oid_name).intersection(old_oid_name)
display_name_intersection = set(new_oid_display_name).intersection(old_oid_display_name)
print(len(name_intersection),len(display_name_intersection))
if((set(new_oid_name) | name_intersection) == set(new_oid_name)):
    print("New OID contains all the name from old OID")
else:
    print("you got screwed")
    
if((set(new_oid_display_name) | display_name_intersection) == set(new_oid_display_name)):
    print("New OID contains all the display name from old OID")
else:
    print("you got screwed again")    
difference_nvo_name = set(new_oid_name) - set(old_oid_name)
print(len(difference_nvo_name))
difference_nvo_display_name = set(new_oid_display_name) - set(old_oid_display_name)
print(len(difference_nvo_display_name))
diff_name = new_oid[new_oid['name'].isin(list(difference_nvo_name))]
diff_name
diff_name.shape
lower_new_oid_display_name = new_oid['display_name'].str.lower()
diff_display_name = new_oid[lower_new_oid_display_name.isin(list(difference_nvo_display_name))]
diff_display_name
difference_nvo_display_name = set([w.capitalize() for w in list(difference_nvo_display_name)])

renamed_classes = pd.concat([diff_name,diff_display_name]).drop_duplicates(keep=False)
renamed_classes
d=[[x, y.capitalize()] for x,y in zip(old_oid_name,old_oid_display_name)]
d = np.array(d)
old_oid = pd.DataFrame(data = d, columns = ['name','display_name'])
old_oid.head()
old_oid.loc[old_oid['name'].isin(list(renamed_classes['name']))]
for x,y in zip(list(renamed_classes['name']),list(renamed_classes['display_name'])):
    print(x,y)
    old_oid.loc[old_oid.name == x , 'display_name'] = y
old_oid.loc[old_oid['name'].isin(list(renamed_classes['name']))]
new_labels = new_oid[new_oid.name.isin(list(difference_nvo_name))]
new_labels
new_labels.shape