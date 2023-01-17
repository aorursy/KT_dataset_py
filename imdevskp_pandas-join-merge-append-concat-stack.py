import pandas as pd
# df_a.append?
df_a = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

print(df_a, '\n')



df_b = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

print(df_b, '\n')



print(df_a.append(df_b, ignore_index=True))
# pd.concat?
df_a = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

print(df_a, '\n')



df_b = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

print(df_b, '\n')



df_c = pd.DataFrame([[9, 10], [11, 12]], columns=list('AB'))

print(df_c, '\n')



print(pd.concat([df_a, df_b, df_c], axis=0), '\n') # default



print(pd.concat([df_a, df_b, df_c], axis=1))
# df.join?
df_a = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 2, 6]})

print(df_a, '\n')



df_b = pd.DataFrame({'A': [1, 2, 3, 6], 'C': ['a', 'b', 'c', 'd']})

print(df_b, '\n')



df_c = pd.DataFrame({'A': [1, 4, 5, 6], 'D': ['xg', 'dt', 'qh', 'yw']})

print(df_b, '\n')



print(df_a.set_index('A').join([df_b.set_index('A'), df_c.set_index('A')], 

                               how='outer'))
# pd.merge?
df_a = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 2, 6]})

print(df_a, '\n')



df_b = pd.DataFrame({'A': [2, 3, 1, 6], 'C': ['a', 'b', 'c', 'd']})

print(df_b, '\n')



print(pd.merge(df_a, df_b), '\n') # automatically finds the common colum



print(pd.merge(df_a, df_b, how='outer'))