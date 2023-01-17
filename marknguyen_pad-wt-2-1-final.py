# Create lists of numbers

nums = [22,21,12,49,13,63,59]

nums2 = [22,1,4,8,9,10,-1,-4,6,5,9,10,100,-3,-5,8,59,-4,5,2,-1,9,49,5,9,3,22,11,9,0,4,9,10,55,9,10,-2,4,8,9,119,128,92,-2,-48,48,28,29,48,45,76,94,11,0,2]
## Calculate the average of nums using a for loop

avg = 0

for num in nums:

    avg = avg + num

avg = avg / len(nums)

print("Avg: {:.2f}".format(avg))
## Calculate the average of nums using index based looping

avg = 0

for i in range(len(nums)):

    avg = avg + nums[i]

avg = avg / len(nums)

print("Avg: {:.2f}".format(avg))
## Calculate standard deviation of nums using a loop

std = 0

for num in nums:

    std = std + (num - avg)**2

std = std * 1/len(nums)

std = std ** (0.5)

print("Standard deviation: {0:.2f}".format(std))
## Find the first element in num2s where its previous two elements in the sequence add up to this element. Use a while loop.



### Declare tracker variables to track our progress during every iteration

exit_loop = False

i = 2



while (not exit_loop):

    e1 = nums2[i - 2]

    e2 = nums2[i - 1]

    cur_e = nums2[i]

    

    if (e1 + e2 == nums2[i]):

        print("Found the element at index: {:d}, element is: {:d}".format(i,cur_e))

        exit_loop = True

    else:

        i = i + 1

        

    if i >= len(nums2):

        print("Cannot find any element")

        exit_loop = True