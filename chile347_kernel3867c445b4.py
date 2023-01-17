print('if you only have 1 , 2 , 3 , or 4 items put 1 for a blank / max is 5')
calories_1 = int(input())
calories_2 = int(input())
calories_3 = int(input())
calories_4 = int(input())
calories_5 = int(input())
if calories_5 == 1:
    mean = int((calories_1 + calories_2 + calories_3 + calories_4) / 4)
    print(mean)
if calories_4 == 1:
    mean = int((calories_1 + calories_2 + calories_3 + calories_5) / 4)
    print(mean)
if calories_3 == 1:
    mean = int((calories_1 + calories_2 + calories_5 + calories_4) / 4)
    print(mean)
if calories_2 == 1:
    mean = int((calories_1 + calories_5 + calories_3 + calories_4) / 4)
    print(mean)
if calories_1 == 1:
    mean = int((calories_5 + calories_2 + calories_3 + calories_4) / 4)
    print(mean)
if calories_1 + calories_2 == 2:
    mean = int((calories_3 + calories_4 + calories_5) / 3)
    print(mean)
if calories_2 + calories_3 == 2:
    mean = int((calories_1 + calories_4 + calories_5) / 3)
    print(mean)
if calories_3 + calories_4 == 2:
    mean = int((calories_1 + calories_2 + calories_5) / 3)
    print(mean)
if calories_4 + calories_5 == 2:
    mean = int((calories_3 + calories_1 + calories_2) / 3)
    print(mean) 
else:
    mean = int((calories_1 + calories_2 + calories_3 + calories_4 + calories_5) / 5)
    print(mean)
print('one of them is the right mean')#theres a problem with the devsion but one of them is correct for sure

