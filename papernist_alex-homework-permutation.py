# Prog-09: Permutation
# Fill in your ID & Name
# ...
# Declare that you do this by yourself ..... R u sure?
import math
def order_of(permutation):
    perm = permutation.copy()
    order = 1
    while len(perm) != 0:
        number = perm[0]
        length = len(perm) - 1
        order += math.factorial(length) * (number-1)
        perm.pop(0)
        perm = [j-1 if j > number else j for j in perm]
    return order

order_of([2,3,1,4])
def permutation_at(order, n):
    order -= 1
    # order start with digit 0 in computer
    number_in_list = list(range(1,n+1))
    list_number = []
    while len(list_number) != n:
        number = math.floor(order / math.factorial(n-len(list_number)-1))
        # find number with factorial such as first digit [3,x,x,x,x] come from 4! = 24 and 52 / 24 = 2 
        # then find nubmer 2 in number_in_list position 2 in list is 3
        # number_in_list = [1,2,3,4,5] like this
        order %= math.factorial(n-len(list_number)-1)
        # then order %=24 because we need to go next
        # dont stay in the past
        list_number.append(number_in_list[number])
        # append the list
        number_in_list.pop(number)
        # pop number that was used
    return list_number

permutation_at(52, 5)
def next_permutation(permutation):
    if order_of(permutation) == math.factorial(len(permutation)):
        return 'None'
    else:
        return permutation_at(order_of(permutation)+1, len(permutation))

next_permutation([1,5,6,2,4,3])
def prev_permutation(permutation):
    if order_of(permutation) == 1:
        return 'None'
    else:
        return permutation_at(order_of(permutation)-1, len(permutation))

prev_permutation([1, 5, 6, 3, 2, 4])
def longest_cycles(permutation):
    perm = permutation.copy()
    check_perm = perm.copy()
    new_list = []
    # preprocessing of while loop
    number = perm[0]
    new_list.append([])
    while len(perm) != 0:
        if number not in perm: # append new list if number is not in loop
            new_list.append([])
            number = perm[0]
        new_list[-1].append(number) # append list
        perm.remove(number) # remove used
        number = check_perm[number-1] # next number
    
    max_len = max([len(i) for i in new_list])
    ans_list = []
    for i in new_list:
        if len(i) == max_len:
            ans_list.append(i)
            
    return ans_list

longest_cycles([9 ,6, 2, 11, 12, 10, 8 ,7 ,5, 3, 4, 1])
def main():
    while True:
        x = input().split()
        cmd = x[0]
        p = [int(e) for e in x[1:]]
        if cmd == 'O':
            print(order_of(p))
        elif cmd == 'A':
            print(permutation_at(p[0], p[1]))
        elif cmd == 'N':
            print(next_permutation(p))
        elif cmd == 'P':
            print(prev_permutation(p))
        elif cmd == 'C':
            print(longest_cycles(p))
        elif cmd == 'Q':
            return
#-------------------------------------
main()
