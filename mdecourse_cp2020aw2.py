#sum = 0
# 1 + 2 + 3 + 4 ..... = ?
# iterator
# index
# keywords
# indentation 縮排 4 個 spaces
# num = 100
# define
# 函式 function
def addTo(num):
    sum = 0
    for i in range(1,num+1):
        #print(i)
        sum = sum + i
    return sum

sum = addTo(33)
print(sum)