num = 0

next_num = 1

total_even_fib = 0



while True:

    num, next_num = next_num, num + next_num

    if next_num >= 4000000:

        break

    if next_num % 2 == 0:

        total_even_fib += next_num

        

print(total_even_fib)