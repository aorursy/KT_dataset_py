# 1a - Knapsack 0-1 dengan pendekatan Dynamic Programming (DP) dalam waktu 𝑂(𝑛𝑊)



# Input:

# W - kapasitas kantong (snapsack)

# weight - array yang masing-masing elemennya mewakili berat terkait dengan n item

# value - array yang masing-masing elemennya mewakili harga terkait dengan n item

# Output: Total harga barang maksimal yang bisa didapatkan



def knapSack(W, weight, value):

    n = len(value)

    K = [[0 for x in range(W+1)] for x in range(n+1)] 

    

    for i in range(n+1): 

        for w in range(W+1): 

            if i==0 or w==0: 

                K[i][w] = 0

            elif weight[i-1] <= w: 

                K[i][w] = max(value[i-1] + K[i-1][w-weight[i-1]],  K[i-1][w]) 

            else: 

                K[i][w] = K[i-1][w] 

  

    return K[n][W]



# Contoh input:

W = 50

weight = [10, 20, 30]

value = [60, 100, 120]



# Hasil yang diharapkan:

# 220, hasil penjumlahan value 100 + 120

print("Total harga barang maksimal yang bisa didapatkan adalah", knapSack(W, weight, value)) 
# Nomor 2 - Longest Common Increasing Subsequence (LCIS)



# Input: Dua buah array arr1 dan arr2 yang berisi bilangan bulat

# Output: Panjang dari LCIS



def LCIS(arr1, arr2):

    n = len(arr1) 

    m = len(arr2)  

    table = [0] * m

    result = 0

    

    for j in range(m): 

        table[j] = 0



    for i in range(n):

        current = 0

        for j in range(m): 

            if (arr1[i] == arr2[j]): 

                if (current + 1 > table[j]): 

                    table[j] = current + 1

            if (arr1[i] > arr2[j]): 

                if (table[j] > current): 

                    current = table[j]

  

    for i in range(m): 

        if (table[i] > result): 

            result = table[i] 

  

    return result



# Contoh input:

arr1 =  [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]

arr2 = [1, 4, 1, 4, 2, 1, 3, 5, 6, 2, 3, 7, 3, 0, 9, 5]



# Hasil yang diharapkan:

# 6, yang merupakan panjang dari [1, 4, 5, 6, 7, 9]

print("Panjang dari LCIS adalah", LCIS(arr1, arr2))
# Nomor 3 - Longest Non-Overlapping Reverse Substring



# Input: Sebuah string str

# Output: Panjang substring terpanjang yang susunan asli dan susunan kebalikannya

# muncul pada string str tanpa ada karakter yang overlap



def longestNonOverlappingReverse(str):

    longest = 0



    for i in range(len(str)):

        for j in range(i + 1, len(str) + 1):

            subString = str[i:j] 

            reversedSubstring = subString[::-1]

            length = len(subString)

            

            if reversedSubstring in str:

                if str.find(reversedSubstring, j) != -1:

                    if length > longest:

                        longest = length



    return longest



# Contoh input:

s = "RECURSE"



# Hasil yang diharapkan:

# 2, untuk "RE" dan "ER"

print("Panjang substring adalah", longestNonOverlappingReverse(s))