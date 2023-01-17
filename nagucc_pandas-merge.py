import pandas as pd
#定义资料集并打印出

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                             'A': ['A0', 'A1', 'A2', 'A3'],

                             'B': ['B0', 'B1', 'B2', 'B3']})

print(left)
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],

                              'C': ['C0', 'C1', 'C2', 'C3'],

                              'D': ['D0', 'D1', 'D2', 'D3']})

print(right)
#依据key column合并，并打印出

res = pd.merge(left, right, on='key')



print(res)
left2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],

                      'key2': ['K0', 'K1', 'K0', 'K1'],

                      'A': ['A0', 'A1', 'A2', 'A3'],

                      'B': ['B0', 'B1', 'B2', 'B3']})

print(left2)
right2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],

                       'key2': ['K0', 'K0', 'K0', 'K0'],

                       'C': ['C0', 'C1', 'C2', 'C3'],

                       'D': ['D0', 'D1', 'D2', 'D3']})

print(right2)
#依据key1与key2 columns进行合并，使用inner，只保留两个key完全相同，且存在的数据

res = pd.merge(left2, right2, on=['key1', 'key2'], how='inner')

print(res)
# 使用outer，不存在的数据会设置为NaN

res = pd.merge(left2, right2, on=['key1', 'key2'], how='outer')

print(res)
# left，保留所有left的行

res = pd.merge(left2, right2, on=['key1', 'key2'], how='left')

print(res)
# right，保留所有right的行

res = pd.merge(left2, right2, on=['key1', 'key2'], how='right')

print(res)