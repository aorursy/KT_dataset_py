thisdict={'x':10,'y':15,'z':20}
print("AN ITEM : %d" %thisdict['x'])
mydict={'x':4}
print("AN ITEM : %d" %mydict['x'])
print("THE KEYS: %s" %thisdict.keys())
print("THE VALUES: %s" %thisdict.values())
for i in thisdict.keys():
    print (thisdict[i])