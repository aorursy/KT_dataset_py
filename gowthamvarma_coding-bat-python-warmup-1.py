def sleepIn(weekday,vacation) :
    # your code here
    return False
print('Test cases ::')
print('')
print ('sleepIn(False, False) == True ')
print ('Output : ' , sleepIn(False, False))
print("#Pass") if sleepIn(False, False) == True else print("#Fail")
print('')
print ('sleepIn(True, False) == False ')
print ('Output : ' , sleepIn(True, False))
print("#Pass") if sleepIn(True, False) == False else print("#Fail")
print('')
print ('sleepIn(False, True) == True ')
print ('Output : ' , sleepIn(False, True))
print("#Pass") if sleepIn(False, True) == True else print("#Fail")

# solution
def sleepIn(weekday,vacation) :
    return (not weekday) | vacation
def monkeyTrouble(aSmile,bSmile):
    # your code here
    return False
print('Test cases ::')
print('')
print ('monkeyTrouble(True, True) == True ')
print ('Output : ' , monkeyTrouble(True, True))
print("#Pass") if monkeyTrouble(True, True) == True else print("#Fail") 
print('')
print ('monkeyTrouble(False, False) == True ')
print ('Output : ' , monkeyTrouble(False, False))
print("#Pass") if monkeyTrouble(False, False) == True else print("#Fail")
print('')
print ('monkeyTrouble(True, False) == False ')
print ('Output : ' , monkeyTrouble(True, False))
print("#Pass") if monkeyTrouble(True, False) == False else print("#Fail")
#solution
def monkeyTrouble(aSmile,bSmile):
  if (aSmile & bSmile):
    return True
  if ((not aSmile) & ( not bSmile)):
    return True
  return False







