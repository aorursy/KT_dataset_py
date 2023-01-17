import csv

import random



# To ensure every run is the same

random.seed(42)



# Open input and skip the header. Print it to sanity check.

with open('../input/test.csv') as test_file:

    reader = csv.reader(test_file)

    print(next(reader))



    # Open the output file in write mode

    with open("baseline_model.csv", "w") as model_file:

        writer = csv.writer(model_file)

        writer.writerow(["PassengerId", "Survived"])	# write the column headers

        num_rows = 0

        num_heads = 0

        

        # Simply loop over test input to get all the passenger IDs, and toss a coin

        for row in reader:

            coin_toss = random.randint(0, 1)

            

            # Keep track of stats so we can show the result later

            num_rows += 1

            num_heads += coin_toss

            newrow = [row[0], str(coin_toss)]

            writer.writerow(newrow)

            

print('Wrote baseline_model.csv, with %d heads and %d tails' % (num_heads, num_rows-num_heads))