import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import fpgrowth

from mlxtend.frequent_patterns import association_rules



dataset = pd.read_csv('../input/amostra-cadunico-belo-horizonte-2017/amostra_bh_set.tab', sep=' ', header=None)

dataset = dataset.values.tolist()
te = TransactionEncoder()

te_ary = te.fit(dataset).transform(dataset)

df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)

frequent_itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

to_plot = rules[ 

#        (rules['antecedent_len'] >= 3) &

#        (rules['consequent_len'] == 1) &

       (rules['support'] > 0.7) &

       (rules['confidence'] > 0.9) &

       (rules['lift'] > 1.1) ]
def print_point(df,i):

    print("%.3f, %.3f, %.3f, %.3f, \"%s => %s\""%(

        to_plot.iloc[i,:].support,

        to_plot.iloc[i,:].confidence,

        to_plot.iloc[i,:].conviction,

        to_plot.iloc[i,:].lift,

        list(to_plot.iloc[i,:].antecedents),

        list(to_plot.iloc[i,:].consequents)))



print("support,confidence,conviction,lift,rule")

fig, (conf_rsup,conv_lift) = plt.subplots(1,2,figsize=(15,5))





# conf_rsup = fig.add_subplot(1,3,1)



x = to_plot['support'].values.tolist()

y = to_plot['confidence'].values.tolist()



conf_rsup.scatter(x, y,s=5,label='relative support X Confidence')

conf_rsup.set_xlabel("rsup")

conf_rsup.set_ylabel("conf")

conf_rsup.grid(True)

conf_rsup.set_xlim([0.76, 0.8])

conf_rsup.set_ylim([0.90, 0.96])



for i, txt in enumerate(x):

    if(txt > 0.795):

        conf_rsup.annotate("(%.3f,%.3f)"%(txt,y[i]),(x[i],y[i]))

        print_point(to_plot,i)



x1 = to_plot['conviction'].values.tolist()

y1 = to_plot['lift'].values.tolist()



conv_lift.scatter(x1, y1, s=5, label='Conviction X Lift')

conv_lift.set_xlabel("conv")

conv_lift.set_ylabel("lift")

conv_lift.grid(True)

conv_lift.set_xlim([1.8, 3.4])

conv_lift.set_ylim([1.099, 1.12])



for i, txt in enumerate(x1):

    if(txt > 3.38):

        conv_lift.annotate("(%.3f,%.3f)"%(txt,y1[i]),(x1[i],y1[i]))

        print_point(to_plot,i)



plt.show()