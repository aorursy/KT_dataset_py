import pandas as pd
tri_raw = pd.read_csv('../input/euler18.csv', header=None)
tri_raw
tri_lst = []

for row in range(len(tri_raw)):
    tri_lst.append(tri_raw.iloc[row,0])

tri_lst
tri_str = [el.split() for el in tri_lst]
tri_int = []

for i in range(len(tri_str)):
    tri_int.append([int(el) for el in tri_str[i]])
    
tri_int
# row_15 = tri_int[14]
# row_14 = tri_int[13]

# compare element 0 in row 14 with element 0 and 1 in row 15
# compare element 1 in row 14 with element 1 and 2 in row 15
# ...

tri_cum = tri_int

# for each row in the triangle, starting from the second to last
for row in reversed(range(len(tri_cum)-1)):  
    # for each element in this row
    for el in range(len(tri_cum[row])):  
        # add the highest adjacent number to create a maximum cumulative sum for that point
        tri_cum[row][el] += max(tri_cum[row+1][el], tri_cum[row+1][el+1])
tri_cum