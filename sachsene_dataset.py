import glob

all_scrapped_files = glob.glob('/kaggle/input/amazons-advertisements/scrapped_data/scrapped_data/*/*.csv')

for i in all_scrapped_files:

    count = 0

    with open(i, 'r') as file:

        for line in file:

            count += 1

print('Amount of scrapped files: ', len(all_scrapped_files))
for i in range(len(all_scrapped_files)):

    with open(all_scrapped_files[i], 'r') as file:

        if i == 0:

            data = file.read()

        else:

            ad = file.read(3)

            data = file.read()

        print('Reading {n} '.format(n=all_scrapped_files[i]))

        with open('amazon_combined_scrapped_data.csv', 'a') as output_file:

            print('Writting {n} '.format(n=all_scrapped_files[i]))

            output_file.write(data)