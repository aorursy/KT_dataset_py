import pandas as pd



df = pd.read_csv('../input/clicks.csv')

df.head(20)
df["is_purchase"] = df["click_day"].apply(lambda x: "Purchase" if pd.notnull(x) else "No Purchase")

purchase_counts = df.groupby(["group", "is_purchase"]).user_id.count().reset_index()

print(purchase_counts)
from scipy.stats import chi2_contingency



# contingency = [[A_purchases, A_not_purchases],

#                [B_purchases, B_not_purchases],

#                [C_purchases, C_not_purchases]]



contingency = [[316, 1350],

               [183, 1483],

               [83, 1583]]



pvalue = chi2_contingency(contingency)[1]



f = lambda x: True if x <= 0.05 else False



is_significant = f(pvalue)



print(is_significant)
# calculate the percent of visitors who would need to purchase the upgrade package at each price point ($0.99, $1.99, $4.99) in order to generate sales target of $1,000 per week.



num_visits = len(df)



p_clicks_099 = (1000 / 0.99) / num_visits

p_clicks_199 = (1000 / 1.99) / num_visits

p_clicks_499 = (1000 / 4.99) / num_visits
from scipy.stats import binom_test



x = 316

n = 1350 + 316

p = p_clicks_099



pvalueA = binom_test(x, n, p)



x = 183

n = 1483 + 183

p = p_clicks_199



pvalueB = binom_test(x, n, p)



x = 83

n = 1583 + 83

p = p_clicks_499



pvalueC = binom_test(x, n, p)



print(pvalueA, pvalueB, pvalueC)

print("Intuitively we may want to choose $0.99 as it is likely to be accepted by users.\nSurprisingly, $4.99 should be our choice.")