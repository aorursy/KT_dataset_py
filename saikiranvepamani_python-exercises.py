7 ** 4
s = "Hi there Sam!"
s.split()
planet = "Earth"
diameter = 12742
print('the diamter of {} is {} kilometers'.format(planet,diameter))
lst = [1,2,[3,4],[5,[100,200,['hello']],23,11],1,7]
lst[3][1][2][0]
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
d['k1'][3]['tricky'][3]['target'][3]
# Tuple is immutable
def domainGet(str):
    return str.split('@')[1]
domainGet('user@domain.com')
def findDog(word):
    return 'dog' in word.lower().split()
findDog('Is there a dog here?')
def countDog(string):
    count = 0
    for word in string.lower().split():
        if word == 'dog':
            count += 1
    return count
countDog('This dog runs faster than the other dog dude!')
seq = ['soup','dog','salad','cat','great']
list(filter(lambda x: x.startswith('s'),seq ))
def caught_speeding(speed, is_birthday):
        
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed
    
    if speeding > 80:
        return 'Big Ticket'
    elif speeding > 60:
        return 'Small Ticket'
    else:
        return 'No Ticket'
caught_speeding(81,True)
caught_speeding(81,False)