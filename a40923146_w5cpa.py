''' now can we read [[40727216, 40727219, 40923113, 40923114, 40923116, 40923121, 40923129, 40923137, 40923140, 40923146, 40923147]] 
into one dimensional list
'''
cp1aGroup = [[40727216, 40727219, 40923113, 40923114, 40923116, 40923121, 40923129, 40923137, 40923140, 40923146, 40923147]] 
# len() can be used to get the length of a list
#print(len(cp1aGroup))
# so we can use the for loop to read group member out
groupNum = len(cp1aGroup)
cp1a = []
for i in range(groupNum):
    # use len() to get student number for each group
    studNum = len(cp1aGroup[i])
    #print(cp1aGroup[i])
    for j in range(studNum):
        cp1a.append(cp1aGroup[i][j])
print(cp1a)
# yes, we transfer two dimensional list into one diimension
# for the next step we may need to compare two lists to find the discrepancy
