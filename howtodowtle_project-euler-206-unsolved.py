from math import sqrt
import pandas as pd
def create_no(i):
    """
    Returns the integer 1_2_3_4_5_6_7_8_9_0, where _ is
    a single digit of value i.
    """
    str_list = list("1_2_3_4_5_6_7_8_9_0")
    i_list = [str(i) if el=="_" else el for el in str_list]
    i_number = int(''.join(i_list))
    return i_number
lower = int(sqrt(1020304050607080900)/10)*10
upper = int(sqrt(1929394959697989990)/10)*10+10
lower, upper
(upper-lower)/10
(upper-lower)/100*2
try_30 = [i*100+30 for i in range(1,11)]
try_70 = [i*100+70 for i in range(1,11)]
try_both = sorted(try_30 + try_70)
pd.DataFrame([j*j for j in try_both], index=try_both)
sqrt(184900), sqrt (280900), sqrt(688900)
(upper-lower)/1000*3
try_430 = [i*1000+430 for i in range(1,10)]
try_530 = [i*1000+530 for i in range(1,10)]
try_830 = [i*1000+830 for i in range(1,10)]
try_all = sorted(try_430 + try_530 + try_830)
pd.DataFrame([j*j for j in try_all], try_all)
def check_fit(number, string="1_2_3_4_5_6_7_8_9_0"):
    """
    Returns True if number matches int(string)
    at every even index position.
    """
    
    str_list = list("1_2_3_4_5_6_7_8_9_0")
    num_list = list(str(number**2))
    if len(num_list) < len(str_list)-1:
        return False
    else:
        bools = []
        for el in range(0, len(str_list), 2):
            bools.append(str_list[el] == num_list[el])
        if sum(bools) == len(bools):
            return True
        else:
            return False
# check every 30 and 70
lower = int(sqrt(1020304050607080900)/100)*100-30
upper = int(sqrt(1929394959697989990)/100)*100+100+30
lower, upper
test = lower
# 
while test <= upper:
    if check_fit(test):
        print(f"Yeah! {test}^2 = {test**2}!")
        break
    test += 60  # from ...70 to ...30
    if check_fit(test):
        print(f"Yeah! {test}^2 = {test**2}!")
        break
    test += 40  # from ...30 to ...70