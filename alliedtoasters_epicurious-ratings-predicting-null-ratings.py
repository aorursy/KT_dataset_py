import numpy as np

import pandas as pd

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

import datetime

from IPython.display import display

from IPython.display import HTML

import IPython.core.display as di 

%matplotlib inline



htmlCode="""

<style>

    .button {

        background-color: #008CBA;;

        border: none;

        color: white;

        padding: 8px 22px;

        text-align: center;

        text-decoration: none;

        display: inline-block;

        font-size: 16px;

        margin: 4px 2px;

        cursor: pointer;

    }

</style>

<script>

    code_show=true;

    function code_toggle() {

        if (code_show){

            $('div.input').hide();

        } else {

            $('div.input').show();

        }

        code_show = !code_show

    }

    $( document ).ready(code_toggle);

</script>



"""

HTML(htmlCode)
raw_data = pd.read_json('../input/full_format_recipes.json').dropna(thresh=2)  #Dropping some null rows

raw_data = raw_data[raw_data.rating.notnull()]  #Drop rows with null ratings



raw_data.rating.hist(width=.4);

plt.title('Distribution of Ratings');
#Define Outcome Variable

raw_data['no_rating'] = np.where(raw_data.rating==0, 1, 0)

raw_data['four_plus'] = np.where(raw_data.rating>4, 1, 0)
latest_date = raw_data.date.max()

raw_data['t_delta'] = latest_date-raw_data.date

raw_data['years_old'] = raw_data.t_delta.apply(lambda x: int(x.days/365))

raw_data['age'] = raw_data.years_old/raw_data.years_old.max()



#Drop the 13 recipes with strange dates, like 20+ years old recipes.

raw_data = raw_data[raw_data.years_old <= 12]



raw_data['half_years'] = raw_data.t_delta.apply(lambda x: int(x.days/182))
hyrs = pd.Series(raw_data.half_years.unique()).sort_values()

mean_rating = []

four_plus = []

n_recipes = []

zeros = []

for hy in hyrs:

    df = raw_data[raw_data.half_years == hy]

    mean_rating.append(df.rating.mean())

    four_plus.append(df.four_plus.sum())

    zeros.append(df.no_rating.sum())

    n_recipes.append(len(df))

    

n_recipes = pd.Series(n_recipes)

perc_zeros = zeros/n_recipes

four_plus = four_plus/n_recipes

n_recipes = n_recipes/n_recipes.max()

mean_rating = np.array(mean_rating)/5



    

sns.set_style('whitegrid')

fig = plt.figure(figsize=(7, 5));

years = np.array(hyrs)/2

plt.plot(years, mean_rating, label='mean rating (scaled from 0 to 1)');

plt.plot(years, n_recipes, label='number of recipes published');

plt.plot(years, perc_zeros, label='proportion of zero ratings');

plt.plot(years, four_plus, label='proportion of 4+ ratings');

plt.title('Evolution of Ratings Over Time');

plt.xlabel('Years Since Publication');

plt.legend(loc=2);
def get_full_dir(lst):

    """consolidates directions from list to single string"""

    doc = ''

    for step in lst:

        doc += step + ' '

    return doc



#Build features...

#Length of ingredients list

raw_data['n_ingredients'] = raw_data.ingredients.apply(len)



#Consolodate directions into single string and get number of words

raw_data['dir_doc'] = raw_data.directions.apply(get_full_dir)

raw_data['n_words'] = raw_data.dir_doc.apply(lambda x: len(x.split()))

raw_data['binned_length'] = np.where(raw_data.n_words <= 50, 1, False)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words <= 100, 2, False), raw_data.binned_length)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words <= 150, 3, False), raw_data.binned_length)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words <= 200, 4, False), raw_data.binned_length)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words <= 300, 5, False), raw_data.binned_length)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words <= 500, 6, False), raw_data.binned_length)

raw_data.binned_length = np.where(raw_data.binned_length == False, np.where(raw_data.n_words > 500, 7, False), raw_data.binned_length)

raw_data['nw_scaled'] = raw_data.n_words/raw_data.n_words.max()

bin_desc = {

    # Maps bin values to length of directions

    1 : '< 50',

    2 : '50-100',

    3 : '100-150',

    4 : '150-200',

    5 : '200-300',

    6 : '300-500',

    7 : '> 500'

}



#Flag whether the recipe has a description

raw_data['has_desc'] = np.where(raw_data.desc, 1, 0)



#Drop the 22 recipes with more than 30 ingredients

raw_data = raw_data[raw_data['n_ingredients'] <= 30]
lengths = pd.Series(raw_data.n_ingredients.unique()).sort_values()

mean_rating = []

rating_std = []

n_recipes = []

four_plus = []

zeros = []



for l in lengths:

    df = raw_data[raw_data.n_ingredients == l]

    four_plus.append(df.four_plus.sum())

    zeros.append(df.no_rating.sum())

    mean_rating.append(df.rating.mean())

    rating_std.append(df.rating.std())

    n_recipes.append(len(df))

    

n_recipes = pd.Series(n_recipes)

four_plus = four_plus/n_recipes

perc_zeros = zeros/n_recipes

n_recipes = n_recipes/n_recipes.max()

mean_rating = np.array(mean_rating)/5

    

sns.set_style('whitegrid')

fig = plt.figure(figsize=(8, 6))

plt.plot(lengths, mean_rating, label='mean rating (0 to 1)');

plt.plot(lengths, n_recipes, label='number of recipes', alpha=.5);

plt.plot(lengths, perc_zeros, label='proportion of zero ratings');

plt.plot(lengths, four_plus, label='proportion of 4+ ratings');

plt.title('Ratings and Number of Ingredients');

plt.xlabel('Number of Ingredients');

plt.ylabel('Rating');

plt.legend(loc=5);
word_bins = pd.Series(raw_data.binned_length.unique()).sort_values()

mean_rating = []

rating_std = []

n_recipes = []

zeros = []

four_plus = []



for w in word_bins:

    df = raw_data[raw_data.binned_length == w]

    zeros.append(df.no_rating.sum())

    four_plus.append(df.four_plus.sum())

    mean_rating.append(df.rating.mean())

    rating_std.append(df.rating.std())

    n_recipes.append(len(df))

    

n_recipes = pd.Series(n_recipes)

four_plus = four_plus/n_recipes

perc_zeros = zeros/n_recipes

n_recipes = n_recipes/n_recipes.max()

mean_rating = np.array(mean_rating)/5

    

sns.set_style('whitegrid')

fig = plt.figure(figsize=(8, 6))

plt.plot(word_bins, mean_rating, label='mean rating (0 to 1)');

plt.plot(word_bins, n_recipes, label='number of recipes', alpha=.5);

plt.plot(word_bins, perc_zeros, label='proportion of zero ratings');

plt.plot(word_bins, four_plus, label='proportion of 4+ ratings');

plt.title('Length of Directions and Ratings');

plt.xlabel('Number of Words in Directions');

plt.ylabel('Rating');

plt.xticks(range(1,8), [bin_desc[i] for i in range(1, 8)]);

plt.legend(loc=0);
def get_categories(cats):

    """takes a list of categories and returns a list of

    meta-categories to help categorize each recipe by type.

    Returns a list.

    Access to categories is available via special inputs.

    if string 'get_keys' is passed as argument, returns list of meta-categories.

    if string 'get_dict' is passed as argument, returns the entire dictionary of

    meta-categories and the categories that belong to them.

    """

    categories = {

        'alcohol': [

            'campari',

            'ginger',

            'eau de vie',

            'vodka',

            'vermouth',

            'digestif',

            'non-alcoholic',

            'pernod',

            'sake',

            'triple sec',

            'chambord',

            'chartreuse',

            'champagne',

            'kirsch',

            'kahlúa',

            'bourbon',

            'marsala',

            'port',

            'sparkling wine',

            'grand marnier',

            'sherry',

            'red wine',

            'white wine',

            'grappa',

            'breadcrumbs',

            'virginia',

            'mezcal',

            'spirit',

            'whiskey',

            'portland',

            'cognac/armagnac',

            'midori',

            'amaretto',

            'bitters',

            'rosé',

            'gin',

            'fortified wine',

            'butterscotch/caramel',

            'calvados',

            'alcoholic',

            'rum',

            'frangelico',

            'scotch',

            'west virginia',

            'liqueur',

            'aperitif',

            'brandy',

            'sangria',

            'cobbler/crumble',

            'wine',

            'tequila'

        ],

        'breakfast': [

            'muffin',

            'crêpe',

            'quiche',

            'breakfast',

            'waffle',

            'omelet',

            'brunch',

            'pancake'

        ],

        'cheese': [

            'cottage cheese',

            'tomatillo',

            'tomato',

            'brie',

            'cheddar',

            'swiss cheese',

            'monterey jack',

            'fontina',

            'gouda',

            'beer',

            'feta',

            'brine',

            'parmesan',

            'mozzarella',

            'marscarpone',

            'blue cheese',

            'cheese',

            'cream cheese',

            'goat cheese',

            'ricotta',

        ],

        'cities': [

            'cambridge',

            'london',

            'kansas city',

            'boston',

            'beverly hills',

            'hollywood',

            'new orleans',

            'miami',

            'providence',

            'yonkers',

            'louisville',

            'st. louis',

            'seattle',

            'healdsburg',

            'san francisco',

            'paris',

            'brooklyn',

            'portland',

            'pasadena',

            'dallas',

            'pittsburgh',

            'aspen',

            'columbus',

            'las vegas',

            'costa mesa',

            'los angeles',

            'denver',

            'washington, d.c.',

            'houston',

            'minneapolis',

            'atlanta',

            'lancaster',

            'long beach',

            'santa monica',

            'chicago'

        ],

        'countries': [

            'italy',

            'georgia',

            'new mexico',

            'jamaica',

            'france',

            'chile',

            'philippines',

            'spain',

            'mexico',

            'chile pepper',

            'egypt',

            'australia',

            'dominican republic',

            'germany',

            'switzerland',

            'canada',

            'ireland',

            'bulgaria',

            'england',

            'guam',

            'japan',

            'israel',

            'haiti',

            'peru'

        ],

        'dairy': [

            'mayonnaise',

            'dairy free',

            'ice cream machine',

            'milk/cream',

            'butterscotch/caramel',

            'peanut butter',

            'egg',

            'egg nog',

            'eggplant',

            'cream cheese',

            'buttermilk',

            'butternut squash',

            'dairy',

            'ice cream',

            'sour cream',

            'butter'

        ],

        'dessert': [

            'frozen dessert',

            'brownie',

            'pot pie',

            '#cakeweek',

            'soufflé/meringue',

            'ice cream machine',

            'dessert',

            'pie',

            'cookie',

            'ice cream',

            'iced coffee',

            'cake',

            'cupcake',

            'candy thermometer',

            'tart',

            'pancake',

            'phyllo/puff pastry dough',

            'candy',

            'coffee grinder',

            'crêpe',

            'cookies',

            'coffee',

            'custard',

            'pastry',

            'sorbet',

            'fritter'

        ],

        'dinner': [

            'salad dressing',

            'casserole/gratin',

            'stuffing/dressing',

            'frittata',

            'chili',

            'potato salad',

            'salad',

            'stock',

            'burrito',

            'pot pie',

            'dinner',

            'hamburger',

            'pizza',

            'soup/stew',

            'tortillas',

            'stew'

        ],

        'drinks': [

            'house cocktail',

            'martini',

            'mixer',

            'steam',

            'coffee grinder',

            'cocktail',

            'punch',

            'spritzer',

            'steak',

            'tea',

            'cocktail party',

            'drink',

            'smoothie',

            'westwood',

            'coffee',

            'iced tea',

            'iced coffee',

            'drinks',

            'margarita',

            'hot drink'

        ],

        'events': [

            'kosher for passover',

            'fall',

            'party',

            'labor day',

            'winter',

            "new year's eve",

            'buffet',

            'fourth of july',

            "new year's day",

            'parade',

            'diwali',

            'oscars',

            'super bowl',

            'thanksgiving',

            'graduation',

            'persian new year',

            'birthday',

            'kentucky derby',

            'spring',

            "valentine's day",

            'hanukkah',

            'potluck',

            'backyard bbq',

            'ramadan',

            'poker/game night',

            'mardi gras',

            'tailgating',

            "st. patrick's day",

            'cocktail party',

            'rosh hashanah/yom kippur',

            'cinco de mayo',

            'halloween',

            'sukkot',

            'kwanzaa',

            "mother's day",

            'flaming hot summer',

            'picnic',

            'pacific palisades',

            'game',

            'shower',

            'wedding',

            'christmas',

            'family reunion',

            'friendsgiving',

            'christmas eve',

            'engagement party',

            'easter',

            'lunar new year',

            'summer',

            'passover',

            'oktoberfest',

            'purim',

            'anniversary',

            'back to school',

            'shavuot',

            'bastille day',

            'camping'

        ],

        'foods': [

            'soy free',

            'dip',

            'vinegar',

            'yogurt',

            'stock',

            'biscuit',

            'pot pie',

            'pizza',

            'omelet',

            'flat bread',

            'tortillas',

            'pickles',

            'muffin',

            'salad dressing',

            'cranberry sauce',

            'casserole/gratin',

            'frittata',

            'salad',

            'food processor',

            'potato salad',

            'sourdough',

            'burrito',

            'breadcrumbs',

            'salsa',

            'soy sauce',

            'sandwich theory',

            'windsor',

            'chili',

            'soy',

            'soup/stew',

            'hummus',

            'stew',

            'bread',

            'stuffing/dressing',

            'seafood',

            'freezer food',

            'sandwich',

            'sauce',

            'hamburger',

            'taco'

        ],

        'fruit': [

            'tangerine',

            'currant',

            'raisin',

            'lemon',

            'fruit juice',

            'quince',

            'apricot',

            'apple juice',

            'honeydew',

            'pineapple',

            'tomato',

            'kiwi',

            'pomegranate juice',

            'lemongrass',

            'melon',

            'tropical fruit',

            'plum',

            'cranberry sauce',

            'fruit',

            'lemon juice',

            'lime juice',

            'passion fruit',

            'guava',

            'asian pear',

            'persimmon',

            'date',

            'cranberry',

            'raspberry',

            'grapefruit',

            'prune',

            'berry',

            'blueberry',

            'dried fruit',

            'orange juice',

            'kumquat',

            'coconut',

            'lychee',

            'blackberry',

            'strawberry',

            'nectarine',

            'watermelon',

            'apple',

            'papaya',

            'cherry',

            'pear',

            'banana',

            'pomegranate',

            'lingonberry',

            'cantaloupe',

            'mango',

            'orange',

            'peach',

            'citrus',

            'lime',

            'fig',

            'grape'

        ],

        'grains': [

            'rice',

            'semolina',

            'brown rice',

            'cornmeal',

            'rye',

            'oatmeal',

            'quinoa',

            'hominy/cornmeal/masa',

            'barley',

            'bulgur',

            'wild rice',

            'bran',

            'oat',

            'goat cheese',

            'grains',

            'granola',

            'whole wheat',

            'corn'

        ],

        'greens': [

            'lettuce',

            'watercress',

            'radicchio',

            'broccoli rabe',

            'broccoli',

            'rutabaga',

            'brussel sprout',

            'mustard greens',

            'spinach',

            'celery',

            'chard',

            'leafy green',

            'cabbage',

            'bok choy',

            'kale'

        ],

        'herbs_spices': [

            'ginger',

            'mint',

            'bell pepper',

            'oregano',

            'spice',

            'poppy',

            'dill',

            'tarragon',

            'caraway',

            'clove',

            'coriander',

            'chile pepper',

            'vanilla',

            'horseradish',

            'saffron',

            'sesame oil',

            'fennel',

            'cumin',

            'herb',

            'curry',

            'parsley',

            'sesame',

            'cinnamon',

            'rosemary',

            'thyme',

            'anise',

            'hot pepper',

            'basil',

            'cilantro',

            'pepper',

            'sage',

            'nutmeg',

            'cardamom',

            'poblano',

            'mustard',

            'paprika'

        ],

        'instructions': [

            'chill',

            'bake',

            'pan-fry',

            'mixer',

            'grill',

            'double boiler',

            'ramekin',

            'boil',

            'juicer',

            'side',

            'stir-fry',

            'wok',

            'blender',

            'slow cooker',

            'skewer',

            'food processor',

            'broil',

            'smoker',

            'deep-fry',

            'grill/barbecue',

            'mandoline',

            'marinate',

            'steam',

            'sauté',

            'microwave',

            'candy thermometer',

            'mortar and pestle',

            'flaming hot summer',

            'shower',

            'roast',

            'freeze/chill',

            'fry',

            'coffee grinder',

            'marinade',

            'pressure cooker',

            'rub',

            'simmer',

            'braise',

            'poach',

            'epi loves the microwave'

        ],

        'jewish': [

            'kosher for passover',

            'rosh hashanah/yom kippur',

            'passover',

            'kosher',

            'purim',

            'hanukkah',

            'sukkot',

            'shavuot'

        ],

        'legumes': [

            'bean',

            'chickpea',

            'tamarind',

            'lentil',

            'sugar snap pea',

            'lima bean',

            'green bean',

            'legume',

            'pea',

            'peanut'

        ],

        'lunch': [

            'mayonnaise',

            'soup/stew',

            'mustard greens',

            'lunch',

            'mustard',

            'pickles'

        ],

        'meat': [

            'beef shank',

            'veal',

            'chambord',

            'duck',

            'bacon',

            'champagne',

            'pork',

            'brisket',

            'poultry',

            'lamb',

            'sausage',

            'rabbit',

            'ground lamb',

            'pork chop',

            'pork tenderloin',

            'lamb shank',

            'ground beef',

            'prosciutto',

            'lamb chop',

            'beef rib',

            'meatloaf',

            'poultry sausage',

            'goose',

            'chicken',

            'venison',

            'steak',

            'rack of lamb',

            'meatball',

            'ham',

            'quail',

            'meat',

            'hamburger',

            'pork rib',

            'beef',

            'beef tenderloin',

            'buffalo'

        ],

        'misc_descrip': [

            'kosher for passover',

            'edible gift',

            'paleo',

            'one-pot meal',

            'weelicious',

            'organic',

            '30 days of groceries',

            'healthy',

            'bon appétit',

            'high fiber',

            '#cakeweek',

            'quick and healthy',

            'condiment/spread',

            'gourmet',

            'condiment',

            'self',

            'tested & improved',

            'harpercollins',

            'bon app��tit',

            'advance prep required',

            'entertaining',

            'kid-friendly',

            'cookbooks',

            'epi + ushg',

            '22-minute meals',

            'cookbook critic',

            '#wasteless',

            'quick & easy',

            'kitchen olympics',

            'house & garden',

            'freezer food',

            'kosher',

            'cook like a diner',

            '3-ingredient recipes',

            'frankenrecipe',

            'kidney friendly',

            'epi loves the microwave',

            'leftovers'

        ],

        'nos': [

            'tree nut free',

            'peanut free',

            'dairy free',

            'soy free',

            'low sodium',

            'sugar conscious',

            'strawberry',

            'vegetarian',

            'wheat/gluten-free',

            'low fat',

            'caraway',

            'low cal',

            'no sugar added',

            'low carb',

            'vegan',

            'no-cook',

            'raw',

            'low sugar',

            'low cholesterol',

            'fat free',

            'pescatarian'

        ],

        'nuts': [

            'tree nut free',

            'hazelnut',

            'chestnut',

            'coconut',

            'seed',

            'peanut butter',

            'butternut squash',

            'macadamia nut',

            'nut',

            'pine nut',

            'almond',

            'cashew',

            'peanut',

            'nutmeg',

            'pistachio',

            'walnut',

            'tree nut'

        ],

        'pasta': [

            'noodle', 

            'couscous', 

            'lasagna', 

            'pasta', 

            'orzo', 

            'pasta maker'

        ],

        'people': [

            'nancy silverton',

            'suzanne goin',

            'emeril lagasse',

            'anthony bourdain'

        ],

        'seafood': [

            'caviar',

            'swordfish',

            'crab',

            'squid',

            'shrimp',

            'shellfish',

            'lobster',

            'bass',

            'cod',

            'oyster',

            'octopus',

            'anchovy',

            'trout',

            'salmon',

            'tilapia',

            'halibut',

            'mussel',

            'snapper',

            'sardine',

            'tuna',

            'seafood',

            'scallop',

            'clam',

            'fish'

        ],

        'snack': [

            'snack',

            'appetizer',

            'dip',

            'hummus',

            'snack week',

            "hors d'oeuvre"

        ],

        'states': [

            'washington',

            'iowa',

            'utah',

            'california',

            'georgia',

            'maryland',

            'illinois',

            'new mexico',

            'idaho',

            'rhode island',

            'maine',

            'nebraska',

            'ohio',

            'tennessee',

            'kansas city',

            'minnesota',

            'south carolina',

            'north carolina',

            'oklahoma',

            'alaska',

            'colorado',

            'arizona',

            'louisiana',

            'kentucky derby',

            'virginia',

            'hawaii',

            'wisconsin',

            'oregon',

            'pennsylvania',

            'new jersey',

            'missouri',

            'michigan',

            'florida',

            'new york',

            'new hampshire',

            'washington, d.c.',

            'connecticut',

            'indiana',

            'vermont',

            'kentucky',

            'west virginia',

            'texas',

            'alabama',

            'mississippi',

            'kansas',

            'massachusetts'

        ],

        'sweets': [

            'sugar conscious',

            'hazelnut',

            'maple syrup',

            'coffee grinder',

            'cr��me de cacao',

            'créme de cacao',

            'honeydew',

            'no sugar added',

            'low/no sugar',

            'honey',

            'jamaica',

            'coffee',

            'sugar snap pea',

            'molasses',

            'jam or jelly',

            'chocolate',

            'iced coffee',

            'low sugar',

            'phyllo/puff pastry dough'

        ],

        'veggies': [

            'yuca',

            'lettuce',

            'shallot',

            'ginger',

            'capers',

            'bell pepper',

            'endive',

            'butternut squash',

            'mushroom',

            'tomatillo',

            'tomato',

            'garlic',

            'arugula',

            'green onion/scallion',

            'cucumber',

            'asparagus',

            'lemongrass',

            'sweet potato/yam',

            'radicchio',

            'watercress',

            'radish',

            'broccoli rabe',

            'okra',

            'squash',

            'leek',

            'escarole',

            'collard greens',

            'potato salad',

            'broccoli',

            'root vegetable',

            'kale',

            'chive',

            'rutabaga',

            'olive',

            'mustard greens',

            'dorie greenspan',

            'zucchini',

            'parsnip',

            'horseradish',

            'fennel',

            'vegetable',

            'potato',

            'pumpkin',

            'turnip',

            'spinach',

            'parsley',

            'artichoke',

            'jícama',

            'beet',

            'avocado',

            'onion',

            'plantain',

            'cauliflower',

            'celery',

            'carrot',

            'cabbage',

            'jerusalem artichoke',

            'yellow squash'

        ]

    }

    if cats == 'get_keys':

        return categories.keys()

    if cats == 'get_dict':

        return categories

    result = []

    for key in categories.keys():

        for cat in cats:

            cat = cat.lower()

            if cat in categories[key]:

                result.append(key)

    return list(set(result))  #remove any duplicates  





raw_data['meta_categories'] = raw_data.categories.apply(get_categories)



def match_cat(recipe_cats, cat):

    """Takes a list of categories from a recipe and a single string,

    checks to see if string in list, if yes, returns 1, else 0.

    """

    if cat in recipe_cats:

        return 1

    return 0



meta_cats = get_categories('get_keys')

meta_cat_features = []

for cat in meta_cats:

    feature = cat + '_meta'

    meta_cat_features.append(feature)

    raw_data[feature] = raw_data.meta_categories.apply(lambda x: match_cat(x, cat))
yr1 = raw_data[raw_data.years_old < 2]



fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(7, 19));

fig.suptitle('Rating Distributions for Recipes in First Two Years of Publication');

ax1.bar(yr1.rating.unique(), yr1.rating.value_counts(), width=.4);

ax1.set_title('All Categories');

df = yr1[yr1['alcohol_meta'] == 1]

ax2.bar(df.rating.unique(), df.rating.value_counts(), width=.4);

ax2.set_title('Alcohol-related recipes');

df = yr1[yr1['events_meta'] == 1]

ax4.bar(df.rating.unique(), df.rating.value_counts(), width=.4);

ax4.set_title('Event-related recipes');

df = yr1[yr1['foods_meta'] == 1]

ax3.bar(df.rating.unique(), df.rating.value_counts(), width=.4);

ax3.set_title('Food-related recipes');
#Random forest classifiers don't really care about data scaling, but I'm going to rescale

#these features now just for consistency.



#It turns out that there is an optimal time "resolution" for model performance

#around two week intervals.

time_scale = 14 #two week timescale

raw_data['time_chunks'] = raw_data.t_delta.apply(lambda x: int(x.days/time_scale))

raw_data['scaled_age'] = raw_data.time_chunks/raw_data.time_chunks.max()

raw_data['n_ingr_sc'] = raw_data.n_ingredients/raw_data.n_ingredients.max()

raw_data['lngth_sc'] = raw_data.binned_length/raw_data.binned_length.max()



scaled_features = [

    'scaled_age',

    'n_ingr_sc',

    'lngth_sc'

]
#feed our features into the random forest classifier

features = meta_cat_features + scaled_features + ['has_desc']

X = raw_data[features]

Y = raw_data['no_rating']

rfc = RandomForestClassifier(n_estimators=200, class_weight='balanced')

rfc.fit(X, Y)
#Let's see what the most important features are

sfm = SelectFromModel(rfc, prefit=True, threshold=.027)

selectors = pd.DataFrame(columns=range(0, len(X.columns)), index=[1, 2])

selectors.loc[1] = selectors.columns

selectors.loc[2] = selectors.columns   #For some reason feature filter needs 2 rows

fts = np.array(selectors)

filtered_features = sfm.transform(selectors)[0]

filtered = []

for ft in list(filtered_features):

    filtered.append(features[ft])

print('{} features selected by random forest model: '.format(len(filtered)), filtered)
sfm = SelectFromModel(rfc, prefit=True, threshold=.006)

selectors = pd.DataFrame(columns=range(0, len(X.columns)), index=[1, 2])

selectors.loc[1] = selectors.columns

selectors.loc[2] = selectors.columns   #For some reason feature filter needs 2 rows

fts = np.array(selectors)

filtered_features = sfm.transform(selectors)[0]

filtered = []

for ft in list(filtered_features):

    filtered.append(features[ft])

print('{} features selected by random forest model: '.format(len(filtered)), filtered)
X = raw_data[filtered]

Y = raw_data['no_rating']

svc = SVC(C=43, gamma = .89, class_weight='balanced')

rslt = cross_val_score(svc, X, Y, cv=5)

print('mean cross-validated accuracy: ', rslt.mean())

print('cross-validation score standard deviation: ', rslt.std())
cutoff = int(.5 * len(raw_data))

train = raw_data[:cutoff]

X_train = train[filtered]

Y_train = train['no_rating']

svc.fit(X_train, Y_train)



test = raw_data[cutoff:]

X_test = test[filtered]

Y_test = test['no_rating']

Y_ = svc.predict(X_test)



tb = pd.crosstab(Y_, Y_test)



print('predictions and true value cross-tabulation:\n', tb)

print('overall accuracy:\n', (tb.iloc[0, 0] + tb.iloc[1, 1]) / tb.sum().sum())
test_imb = 1 - test.no_rating.sum()/len(test)

print('Proportion of nonzero ratings in test set: ', test_imb)