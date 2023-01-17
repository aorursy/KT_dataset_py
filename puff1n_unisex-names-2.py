import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#ls ../input
df_state_names = pd.read_csv('../input/StateNames.csv')
#define function to search for similar names
def GetName(mask_name):
    all_names = df_state_names.Name.unique()
    mask = np.array([mask_name in x for x in all_names])
    names_like = list(all_names[mask])
    filtered = df_state_names[df_state_names.Name.isin(names_like)]
    table = filtered.pivot_table(values='Count', index='Year', columns='Gender', aggfunc=np.sum)
    table = table.div(table.sum(1), axis=0)
    return table, mask_name
find_name = GetName('Alex')
plt.figure(figsize=(10, 6))
plt.plot(find_name[0].F, label='Female', color='red')
plt.plot(find_name[0].M, label='Male', color='blue')

leg = plt.legend(title='Name contain %a as men\'s and woman\'s name' % find_name[1], 
                 loc='best', fontsize='9')
leg.get_frame().set_linewidth(0.0)
plt.plot(find_name[0].index,find_name[0].F,label='Female',color='red') 
plt.plot(find_name[0].index,find_name[0].M,label='Male',color='blue')