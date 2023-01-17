import pandas as pd
df = pd.read_csv("myauto_ge_cars_data.csv")
df.head()
manufacturers = df["Manufacturer"]
manufacturers_dict = {}
for manufacturer in manufacturers:
    if manufacturer in manufacturers_dict:
        manufacturers_dict[manufacturer] += 1
    else:
        manufacturers_dict[manufacturer] = 1
manufacturers_with_counts = []
for key, value in manufacturers_dict.items():
    manufacturers_with_counts.append((key, value))
manufacturers_with_counts = sorted(manufacturers_with_counts, key=lambda y: y[1], reverse=True)
manufacturers_with_counts
manufacturers_list = [elem[0] for elem in manufacturers_with_counts]
# Taking top 10 manufacturers
manufacturers_list = manufacturers_list[:10] 
manufacturers_list