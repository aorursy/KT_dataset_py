def largest_non_adjacent(arr):

    if not arr:

        return 0

    

    return max(

            largest_non_adjacent(arr[1:]),

            arr[0] + largest_non_adjacent(arr[2:]))



largest_sum = largest_non_adjacent([2, 4, 6, 2, 5])

print('largest_sum:', largest_sum )
def largest_non_adjacent(arr):

    if len(arr) <= 2:

        return max(0, max(arr))

    

    cache = [0 for i in arr]

    cache[0] = max(0, arr[0])

    cache[1] = max(cache[0], arr[1])

    

    for i in range(2, len(arr)):

        num = arr[i]

        cache[i] = max(num + cache[i - 2], cache[i - 1])

    return cache[-1]

    

largest_sum = largest_non_adjacent([2, 4, 6, 2, 5])

print('largest_sum:', largest_sum)



'''

arr = [2, 4, 6, 2, 5] --> initialize this given input array



cache = [0 for i in arr] --> initialize the empty cache array with the same lenght as given input array 



#base condition

cache[0] = max(0, arr[0]) --> max(0, 2) = 2

cache[1] = max(cache[0], arr[1]) --> max(2, 4) = 4



#recursive condition

for i in range(2, len(arr)):

    num = arr[i]

    cache[i] = max(num + cache[i - 2], cache[i - 1])

return cache[-1] --> return the right-most element in the cache array



#i = 2 

--> num = arr[2] --> num = 6

--> cache[2] = max(arr[2] + cache[2 - 2], cache[2 - 1])

            --> max(6 + 2, 4) = 8

#i = 3

--> num = arr[3] --> num = 2

--> cache[3] = max(arr[3] + cache[3 - 2], cache[3 - 1]) 

            --> max(2 + 4, 8) = 8

#i = 4

--> num = arr[4] --> num = 5

--> cache[4] = max(arr[4] + cache[2], cache[3])

            --> max(5 + 8, 8) = 13 <== END.

            

==> return cache[-1] --> cache[4] = 13 <== END.           



'''

#This code should run in O(N) and O(N) space. But we can improve this even further.

#Notice that we only ever use the last two elements of the cache when iterating

#through the array --> this sugguests that we could get rid of most of the array

#and just store them as variables.

   
def largest_non_adjacent(arr):

    if len(arr) <= 2:

        return max(0, max(arr))

    

    max_excluding_last = max(0, arr[0])

    max_including_last = max(max_excluding_last, arr[1])

    

    for num in arr[2:]:

        pre_max_including_last = max_including_last

        

        max_including_last = max(max_including_last, max_excluding_last + num)

        max_excluding_last = pre_max_including_last

        

    return max(max_including_last, max_excluding_last)



largest_sum = largest_non_adjacent([2, 4, 6, 2, 5])

print('largest_sum:', largest_sum )

'''

--> arr = [2, 4, 6, 2, 5] --> initialize this given input array

max_ecluding_last = max(0, arr[0]) = 2

max_including_last = max(2, arr[1]) = max(2,4) = 4



for num in arr[2:]:

    pre_max_including_last = max_including_last

    

    max_including_last = max(max_including_last, max_excluding_last + num)

    max_excluding_last = pre_max_including_last



return max(max_including_last, max_excluding_last) 

    

#num = arr[2] = 6

--> pre_max_including_last = max_including_last = 4



*--> max_including_last = max(max_including_last, max_excluding_last + num)

                    --> max(4, 2 + 6) = 8

*--> max_excluding_last = pre_max_including_last = 4



#num = arr[3] = 2

--> pre_max_including_last = 8



*--> max_including_last = max(8, 2 + 4) = 8

*--> max_excluding_last = 8



#num = arr[4] = 5

--> pre_max_including_last = 8



*--> max_including_last = max(8, 8 + 5) = 13

*--> max_excluding_last = 8



==> return max(max_including_last, max_excluding_last)

--> return max(13, 8) = 13 <== END.















'''







arr = [2, 4, 6, 2, 5]

for num in arr[2:]:

    print(num)