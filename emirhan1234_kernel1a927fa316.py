b_words = {"book":"kitap" ,"hurricane" :

 "kasırga","bag" :"çanta" }

print(b_words)

print(type(b_words))
b_bag = b_words["bag"]

print(b_bag)

print(type(b_bag))
#Keys & Values





v_keys = b_words.keys()

v_values = b_words.values()







print(v_keys)

print(type(v_keys))





print()

print(v_values)

print(type(v_values))
b_Number1 = 135

b_Number2 = 90



if b_Number1 > b_Number2:

    print(b_Number1 , " is greater then"

  , b_Number2)

elif b_Number1 < b_Number2:

    print(b_Number1 , " is smaller then"

  , b_Number2)

else :

    print("This 2 variables are equal")
# < , <= , > , >= , == , <>

def b_Total(b_n1 , b_n2):

    if b_n1 > b_n2:

        print(b_n1, " is greater then " 

, b_n2)

    elif b_n1 < b_n2:

        print(b_n1, " is smaller then " 

,b_n2)

    else :

        print("These " , b_n1 , " variab variab les are equal")



            

b_Total(100,70)

b_Total(15,25)

b_Total(20,20)

# using 'IN' with LIST







def f_IncludeOrNot(b_search, b_searchList):

    if b_search in b_searchList :

        print("Good news ! ",b_search ,

" is in list.")

    else :

        print(b_search , " is not in List.Sorry :(")

        

        

l_list = list(b_words.keys())

print(l_list)

print(type(l_list))



f_IncludeOrNot("hurricane" , l_list)

f_IncludeOrNot("rose" , l_list)