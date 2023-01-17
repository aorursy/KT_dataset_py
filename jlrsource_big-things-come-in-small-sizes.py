# Import packages
import matplotlib.pyplot as plt
import nltk
import numpy as np
import sqlite3

# Function definitions
def get_top_100_foods(n):
    """Return a list of data for the top 100 foods for a specific nutrient.
    
    Arguments:
    n (str) -- nutrient
    
    Returns:
    (list of tuples) -- [(rank, nutrient amount, product name),...]
    """
    conn = sqlite3.connect("../input/database.sqlite")
    c = conn.cursor()
    c.execute("SELECT DISTINCT(product_name), " + n + " " +
              "FROM FoodFacts " +
              "WHERE product_name<>'' AND " + n + "<>'' " +
              "AND countries LIKE '%united states%' " +
              "ORDER BY " + n + " DESC")
    result = c.fetchall()
    data = []
    rank = 1
    for i in range(100):
        data.append((rank, result[i][1], result[i][0]))
        rank += 1
    conn.close()
    return data
    
def print_top_100_foods(d, t):
    """Print data for the top 100 foods for a specific nutrient.
    
    Arguments:
    d (list of tuples) -- data [(rank, nutrient amount, product name),...]
    t (str) -- nutrient column title
    """
    print('Rank\t{0}\tProduct Name'.format(t))
    for i in range(len(d)):
        print("{0}\t{1:3.1f}\t{2}".format(d[i][0], d[i][1], d[i][2]))
    print("")

def get_freq_dist_top_10_words(d):
    """Return the top 10 words that occur most frequently in product
    names for a specific nutrient.
    
    Arguments:
    d (list of tuples) -- data [(rank, nutrient amount, product name),...]
    
    Returns:
    (list of lists) -- word frequencies [[word,...], [count,...]]
    """
    s = ""
    for i in range(len(d)):
        s = s + d[i][2] + " "
    tokens = nltk.word_tokenize(s.lower())
    text = nltk.Text(tokens)
    fdist = nltk.FreqDist(text)
    top10 = fdist.most_common(10)
    words=[]
    counts=[]
    for i in range(len(top10)):
        words.append(top10[i][0])
        counts.append(top10[i][1])
    freq = [words, counts]
    return freq

def create_barh_plot(f, n):
    """Create a horizontal bar plot of the top 10 words that occur
    most frequently in product names for a specific nutrient.
    
    Arguments:
    f (list of lists) -- word frequencies [[word,...], [count,...]]
    n (str) -- nutrient
    """
    plt.figure(num=1, figsize=(12,6))
    plt.title('Top 10 words associated with foods high in ' + n, fontsize=16)
    plt.xlabel('frequency', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(np.arange(10.4, 0.4, -1.0), f[0], fontsize=16)
    plt.grid(b=True, which='major', axis='x')
    plt.barh(range(10, 0, -1), f[1], alpha=0.4)
    plt.show()
# Print top 100 foods
carb = get_top_100_foods('carbohydrates_100g')
print_top_100_foods(carb, 'Carbohydrates')
# Plot top 10 words
carb_words = get_freq_dist_top_10_words(carb)
create_barh_plot(carb_words, 'carbohydrates')
# Print top 100 foods
fat = get_top_100_foods('fat_100g')
print_top_100_foods(fat, 'Fat')
# Plot top 10 words
fat_words = get_freq_dist_top_10_words(fat)
create_barh_plot(fat_words, 'fat')
# Print top 100 foods
sodium = get_top_100_foods('sodium_100g')
print_top_100_foods(sodium, 'Sodium')
# Plot top 10 words
sodium_words = get_freq_dist_top_10_words(sodium)
create_barh_plot(sodium_words, 'sodium')
# Print top 100 foods
sugar = get_top_100_foods('sugars_100g')
print_top_100_foods(sugar, 'Sugar')
# Plot top 10 words
sugar_words = get_freq_dist_top_10_words(sugar)
create_barh_plot(sugar_words, 'sugar')
