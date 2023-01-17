list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#forを使用して、list1の内容を表示

for l1 in list1:

    print(l1) 

    

print("\n")    



list2 = ["あああ","いいい","ううう","えええ","おおお"]

#forを使用して、list2の内容を表示

for l2 in list2:

    print(l2)     



print("\n")    



#「range」を使用して0～4を表示

for num1 in range(5):

    print(num1)



print("\n")    



#「range」を使用して3～5を表示

for num2 in range(3, 6):

    print(num2) 



print("\n")    



#「range」を使用して1～5を表示

for num3 in range(1, 6):

    print(num3)



print("\n") 



#「range」を使用して0～9の範囲で2間隔で値を表示

#結果、0～9の範囲で0, 2, 4, 6, 8の値を表示

for num4 in range(0, 10, 2):

    print(num4)   



print("\n")    



#forを使った繰り返し処理で、

#list1のうち、2で割って余りが0のもののみ表示

#または

#list1のうち、偶数のもののみ表示

#list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for l1 in list1:

    if (l1 % 2 == 0):

        print("偶数", l1) 

 

print("\n")  



#forを使った繰り返し処理で、文字列の文字を1文字ずつ表示 



for a1 in "aiueo":

    print(a1)

    

for a2 in "あいうえお":

    print(a2)