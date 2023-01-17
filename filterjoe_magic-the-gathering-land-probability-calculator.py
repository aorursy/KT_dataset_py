import ipywidgets as widgets

from sympy import binomial as combos

from IPython.display import display, HTML
# Ignore this cell (unless you intend to create your own notebooks with messy display output)



# At first all this code was integrated into land_probability() function. It was fewer

# lines of code but harder to read. This HTML and iPython Notebook messiness is not

# important to calculating probability. By pulling out the messy display code, it is

# easier to focus on and understand the probability aspects of land_probability().



# This can be re-used in other iPython/Jupyter Notebooks, simplifing HTML output for interact()



def title_HTML(title):

    return '<h3>' + title + '</h3>'



def intro_HTML(*args):

    lands, cards, drawn_cards, mulligans_so_far = args

    slider_inputs = {'lands: ': lands,

                     'cards: ': cards,

                     'number of cards drawn: ': drawn_cards,

                     'mulligans so far: ': mulligans_so_far}

    s = ''

    for key in slider_inputs:

        s += '<br>' + key + str(slider_inputs[key])

    return s



def start_HTML_table(headers):

    s = '<br><table>\n<tr>'

    for header in headers:

        s += '<th>' + header + '</th>'

    s += '</tr>'

    return s



def row_of_HTML_table(x_int, scipy_float):

    '''

    Args:

      x_int: integer

      scipy_float: float (scipy or regular)

    '''

    return '<tr><td>{0:.>6}</td><td>{1:.2%}</td></tr>'.format(x_int, float(scipy_float))

    # must coerce scipy float type into regular float type for format to work right

    

def finish_HTML_table(s, min_land, max_land, drawn_cards, mulligans_so_far,

                      below_land_prob, above_land_prob, target_land_prob, out_of_range):

    cumulative_p = 1

    for p in out_of_range[:-1]:

        cumulative_p *= p

    s += '</table><br>'

    s += '{0:.1%} chance of between {1} and {2} lands in this {3} card draw.<br>'.format(float(target_land_prob), min_land, max_land, drawn_cards)

    s += '{0:.1%} chance that number of lands will be outside this range.<br><br>'.format(float(1 - target_land_prob))

    s += '{0:.2%} chance that number of lands will be too low after {1} mulligan(s).<br>'.format(float(below_land_prob * cumulative_p), mulligans_so_far)

    s += '{0:.2%} chance that number of lands will be too high after {1} mulligan(s).<br><br>'.format(float(above_land_prob * cumulative_p), mulligans_so_far)

    return s
def land_probability(lands, cards, drawn_cards,

                     min_land, max_land, mulligans=0):

    out_of_range = []

    nonlands = cards - lands

    text = title_HTML("Probabilities for Lands Drawn:")

    for mulligans_so_far in range(mulligans+1):

        text += intro_HTML(lands, cards, drawn_cards - mulligans_so_far, mulligans_so_far)

        text += start_HTML_table(['Lands', 'Probability'])

        target_land_prob, below_land_prob, above_land_prob = 0, 0, 0

        for drawn_lands in range(0, drawn_cards + 1 - mulligans_so_far):

            p = (combos(nonlands, drawn_cards - drawn_lands - mulligans_so_far)

                 * combos(lands, drawn_lands)

                 / float(combos(cards, drawn_cards - mulligans_so_far)))

            if drawn_lands < min_land:

                below_land_prob += p

            elif drawn_lands > max_land:

                above_land_prob += p

            else:

                target_land_prob +=p

            text += row_of_HTML_table(drawn_lands, p)

        out_of_range.append(below_land_prob + above_land_prob)

        text = finish_HTML_table(text, min_land, max_land, drawn_cards - mulligans_so_far, mulligans_so_far,

                                 below_land_prob, above_land_prob, target_land_prob, out_of_range)

    display(HTML(text))
i = widgets.interact(land_probability,

             lands = widgets.IntSlider(min = 1, max = 50, step = 1, value = 24), 

             cards = widgets.IntSlider(min = 1, max = 200, step = 1, value = 60), 

             drawn_cards = widgets.IntSlider(min = 0, max = 20, step = 1, value = 7),

             min_land = widgets.IntSlider(min = 0, max = 3, step = 1, value = 2),

             max_land = widgets.IntSlider(min = 3, max = 7, step = 1, value = 5),

             mulligans = widgets.IntSlider(min = 0, max = 3, step = 1, value = 2)

             )