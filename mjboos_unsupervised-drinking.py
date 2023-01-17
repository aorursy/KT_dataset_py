from json import loads
import pandas as pd
import numpy as np
import re
from os.path import join, dirname
from sklearn.preprocessing import MultiLabelBinarizer
from json import loads
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.manifold import TSNE
import bokeh.plotting as bpl
from bokeh.themes import Theme
from os.path import join, dirname

def preprocess_ingredient(text):
    import unicodedata as ud
    if pd.isna(text):
        return text
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = text.lower()
    text = re.sub(r'[\n\r]', r' ', text)
    if text.isspace():
        text = np.nan
    return text

def preprocess_measure(text):
    import unicodedata as ud
    if pd.isna(text):
        return text
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = text.lower()
    text = re.sub(r'[\n\r]', r' ', text)
    if text.isspace():
        text = np.nan
    return text

def load_data(preprocess=True):
    data = pd.read_csv(join(dirname(__file__), '../input/all_drinks.csv'), encoding='utf-8')
    if preprocess:
        data[['strIngredient{}'.format(i) for i in range(1,16)]] = data[['strIngredient{}'.format(i) for i in range(1,16)]].applymap(preprocess_ingredient)
        data.iloc[:,data.columns.str.contains('Measure')] = data.iloc[:,data.columns.str.contains('Measure')].applymap(preprocess_measure)
    return data

def tokenize_data(data):
    mlb = MultiLabelBinarizer()
    ingredients_series = data[['strIngredient{}'.format(i) for i in range(1,16)]].apply(lambda x : x.dropna().unique().tolist(), axis=1)
    transformed_ingredients = mlb.fit_transform(ingredients_series)
    return transformed_ingredients, mlb

def from_drinks_to_measure(drinks_df, ingredients_vector, mlb):
    '''Expects the original (pre-processed) DataFrame drinks_df,
    the ingredients_vector and MultiLabelBinarizer given by tokenize_data'''
    measures = drinks_df[['strMeasure{}'.format(i) for i in range(1,16)]]
    ingredient_in_df = drinks_df[['strIngredient{}'.format(i) for i in range(1,16)]]
    list_of_transl_dicts = [dict(zip(ingredient_in_df.values[i], measures.values[i])) for i in range(drinks_df.shape[0])]
    measure_vector = np.full(ingredients_vector.shape, '', dtype='object')
    for i in range(measure_vector.shape[0]):
        for j in np.where(ingredients_vector[i])[0]:
            measure_vector[i,j] = list_of_transl_dicts[i][mlb.classes_[j]]
    return measure_vector
def find_ingredient(ingredient, ingredient_names):
    '''This can be edited to allow wildcards for ingredients'''
    return ingredient_names == ingredient

def load_defaults(ingredient_names, default_fn='default_ingredients.csv'):
    '''Loads default ingredients from csv'''
    ingredients = np.genfromtxt(join(dirname(__file__),default_fn), encoding='utf-8', dtype='str', delimiter=',')
    ingredients = [ingredient.strip() for ingredient in ingredients]
    ingredient_indicator = np.zeros(len(ingredient_names))
    for ingredient in ingredients:
        try:
            ingredient_position = np.where(find_ingredient(ingredient, ingredient_names))[0]
            ingredient_indicator[ingredient_position] = 1
        except:
            continue
    return ingredient_indicator

def mask_cocktails(included_ingredients, ingredients_vector):
    cocktails_to_consider = np.logical_not(ingredients_vector[:,included_ingredients==0].any(axis=1))
    masked_recipes = np.ma.masked_array(ingredients_vector, np.logical_or(np.logical_not(cocktails_to_consider)[:,None], np.tile(np.logical_not(included_ingredients)[None], (ingredients_vector.shape[0],1))))
    return masked_recipes

def make_description_string(masked_recipes, measures, ingredient_names):
    description_list = []
    for drink, rec in enumerate(masked_recipes[~masked_recipes.mask.all(axis=1)]):
        drink_string = ''.join(['<br/> <strong>{0}</strong> - {1}'.format(
            ingredient_names[i], measures[~masked_recipes.mask.all(axis=1)][drink,i]) for i in np.where(rec)[0]])
        drink_string = '<p><font size="3">' + drink_string + '</font></p>'
        description_list.append(drink_string)
    return description_list

def make_description_without_measure_string(masked_recipes, ingredient_names):
    return ['<br/>'.join(ingredient_names[np.where(rec)[0]]) for rec in masked_recipes[~masked_recipes.mask.all(axis=1)]]

def make_bokeh_plot(comps, recipe_names, description_strings):
    colors = color_values[~masked_recipes.mask.all(axis=1)]
    data_source=dict(
            colors=colors,
            x=comps[:,0],
            y=comps[:,1],
            images=[picture_urls[i] for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
            desc = recipe_names,
            labels=alcohol_names[~masked_recipes.mask.all(axis=1)],
            ingredients=description_strings
        )
    source = bpl.ColumnDataSource(data_source)
    TOOLTIPS = """
        <div style="border: none !important">
        <div>
            <div>
                <img
                    src="@images" height="150" alt="@images" width="150"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
            <div>
                <br>
                <span style="font-size: 17px; font-weight: bold;">@desc</span>
            </div>
            <div>
            <br>
                <span>@ingredients{safe}</span>
            </div>
        </div>
    """
    p = bpl.figure(tools="reset,pan,wheel_zoom", tooltips=TOOLTIPS)

    p.scatter('x', 'y', marker='o', size=10, fill_color='colors', source=source, color='colors', legend='labels')
    p.axis.visible = False
    p.title.text = "Unsupervised Cocktails"
    p.title.text_color = "white"
    p.title.text_font_size = "40px"
    p.legend.location = "top_left"
    return p, source

color_list = np.array(('#03ed3a', '#e9000b', '#8a2be2', '#00d7ff', '#ffc400'))

drinks = load_data()
picture_urls = drinks.strDrinkThumb.values
lblenc = LabelEncoder()
alcohol_ind = lblenc.fit_transform(drinks['strAlcoholic'].values)
alcohol_names = drinks['strAlcoholic'].values
color_values = color_list[alcohol_ind]
ingredients = pd.Series(drinks[['strIngredient{}'.format(i) for i in range(1,16)]].values.flatten())
ingredient_counts = ingredients.value_counts()
ingredients_vector, mlb = tokenize_data(drinks)
measure_vector = from_drinks_to_measure(drinks, ingredients_vector, mlb)

ingredient_names = np.array([ingredient.lower() for ingredient in mlb.classes_])

# load default cocktail ingredients
included_ingredients = load_defaults(ingredient_names)

def get_masked_recipes_and_names(included_ingredients, ingredients_vector):
    masked_recipes = mask_cocktails(included_ingredients, ingredients_vector)
    selected_recipes = np.logical_not(masked_recipes.mask.all(axis=1))
    recipe_names = drinks['strDrink'].iloc[selected_recipes].values
    return masked_recipes, recipe_names

def make_tsne(masked_recipes, ingredient_names, recipe_names, perplexity=20, early_exaggeration=20, init='pca', random_state=1, **kwargs):
    # perplexity smaller than number of samples gives bad results
    tsne = TSNE(n_components=2, early_exaggeration=early_exaggeration, perplexity=min(perplexity, (~masked_recipes.mask.all(axis=1)).sum()/5), init=init, random_state=random_state)
    comps = tsne.fit_transform(masked_recipes[~masked_recipes.mask.all(axis=1)][:,~masked_recipes.mask.all(axis=0)])
    description_string = make_description_string(masked_recipes, measure_vector, ingredient_names)
    return comps, description_string

from bokeh.models.widgets import CheckboxGroup, Button, Select
from bokeh.layouts import widgetbox, row, layout, column

bpl.output_notebook()
masked_recipes, recipe_names = get_masked_recipes_and_names(included_ingredients, ingredients_vector)

checkbox_group = CheckboxGroup(
        labels=ingredient_names.tolist(), active=[0, 1])
checkbox_group.active = np.where(included_ingredients)[0].tolist()
comps, description_string = make_tsne(masked_recipes, ingredient_names, recipe_names)

tsne_plot, source = make_bokeh_plot(comps, recipe_names, description_string)

def show_selected_cocktail(attrname, old, new):
    find_cocktail = np.where(source.data['desc']==new)[0]
    marker = tsne_plot.select_one({'name' : 'marker'})
    try:
        tsne_plot.renderers.remove(marker)
    except ValueError:
        pass
    tsne_plot.x(source.data['x'][find_cocktail], source.data['y'][find_cocktail], color='white', size=20, name='marker', line_width=2)

select = Select(title='Highlighted recipe', options=recipe_names.tolist())
select.on_change('value', show_selected_cocktail)

def update():
    '''Updates t-SNE plot'''
    new_included_ingredients = np.zeros(len(included_ingredients))
    new_included_ingredients[checkbox_group.active] = 1
    masked_recipes, recipe_names = get_masked_recipes_and_names(new_included_ingredients, ingredients_vector)
    comps, description_string = make_tsne(masked_recipes, ingredient_names, recipe_names)
    colors = color_values[~masked_recipes.mask.all(axis=1)]
    data_dict=dict(
        colors=colors,
        x=comps[:,0],
        y=comps[:,1],
        labels=alcohol_names[~masked_recipes.mask.all(axis=1)],
        images=[picture_urls[i] for i in np.where(~masked_recipes.mask.all(axis=1))[0]],
        desc = recipe_names,
        ingredients=description_string)
    try:
        tsne_plot.renderers.remove(tsne_plot.select_one({'name':'marker'}))
    except ValueError:
        pass
    source.data = data_dict
    select.options = recipe_names.tolist()
    new_selection = select.value if select.value in recipe_names else None
    if new_selection:
        show_selected_cocktail(None,None,new_selection)

update_button = Button(label="Update")
update_button.on_click(update)
widget = widgetbox(update_button, checkbox_group)
widget_select = widgetbox(select)
overall_layout = row(tsne_plot, widget, widget_select)
bpl.curdoc().add_root(overall_layout)