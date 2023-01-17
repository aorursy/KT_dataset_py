import numpy as np
import sys
import matplotlib.pyplot as plt
plt.close('all')
from matplotlib import cm
import scipy.ndimage.filters as filters
class sphere(object):
    def solve_xy(self, a, b, z=0):
        _range = np.arange(-self.range, self.range+self.res, self.res)
        x, y = np.meshgrid(_range, _range)
        c = x**2+y**2+2*a.m/(np.sqrt((x-a.x)**2+(y-a.y)**2+(z-a.z)**2))+\
            2*b.m/(np.sqrt((x-b.x)**2+(y-b.y)**2+(z-b.z)**2))
        print('2d at z=%.2f, c(normalized energy) range is (%.3f, %.3f)'%(z, np.min(c), np.max(c)))
        return x, y, c  #then c is what we saw in the class, function of energy.
    def solve_xz(self, a, b, y=0):
        _range = np.arange(-self.range, self.range+self.res, self.res)
        x, z = np.meshgrid(_range, _range)
        c = x**2+y**2+2*a.m/(np.sqrt((x-a.x)**2+(y-a.y)**2+(z-a.z)**2))+\
            2*b.m/(np.sqrt((x-b.x)**2+(y-b.y)**2+(z-b.z)**2))
        print('2d at y=%.2f, c(normalized energy) range is (%.3f, %.3f)'%(y, np.min(c), np.max(c)))
        return x, z, c  #then c is what we saw in the class, function of energy.
    def solve_L(self, a, b, z=0):
        #solve for Lagrange points.
        _range = np.arange(-self.range, self.range+self.res, self.res)
        x, y = np.meshgrid(_range, _range)
        cx = x-a.m*(x-a.x)/(np.sqrt((x-a.x)**2+(y-a.y)**2+(z-a.z)**2))**3-\
            b.m*(x-b.x)/(np.sqrt((x-b.x)**2+(y-b.y)**2+(z-b.z)**2)**3)
        cy = y-a.m*(y-a.y)/(np.sqrt((x-a.x)**2+(y-a.y)**2+(z-a.z)**2))**3-\
            b.m*(y-b.y)/(np.sqrt((x-b.x)**2+(y-b.y)**2+(z-b.z)**2)**3)
        return cx, cy  #then find where cx=0, cy=0
    def __init__(self, x=0, y=0, z=0, m=0):
        self.x = x
        self.y = y
        self.z = z
        self.m = m
        self.range = 3
            #limitation that x and y will take.
        self.res = 0.002
            #Change this coefficient if there is no lagrange points.
            #Due to different resolution, there might be multi-points at one position.

def potential_plot(miu=0.1, z=0):
    '''
    2d potential energy plot
    '''
    O1 = sphere(x=miu, y=0, z=0, m=1-miu)
        #Here assume distance and mass of objects is normalized. 
        #Set center of mass as original of coordinates.
    O2 = sphere(x=-(1-miu), y=0, z=0, m=miu)
    if z!=0:  
        #if not at z=0, there is no lagrange point.
        print('No Lagrange points. Points on plot means stable motion in one dimension or two, not in z.')
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    
    '''
    x-y plane
    '''
    x, y, c = sphere().solve_xy(O1, O2, z=z)
    contour = ax[0].contourf(x, y, c, np.arange(np.min(c), 20, 0.2), cmap=cm.jet)
        #Change np.arange(np.min(c), 20, 0.2) can give you better plot resolution.
    cb = fig.colorbar(contour, ax=ax[0])
    cb.set_label('Normalized Energy', fontsize=20)
    ax[0].set_aspect(1)
    ax[0].plot([-O1.range, O1.range], [0, 0], c='black')
    ax[0].plot([0, 0], [-O1.range, O1.range], c='black')
    s_m = ax[0].scatter(O1.x, O1.y, s=500*np.sqrt(O1.m), c='magenta', marker='P')
    ax[0].scatter(O2.x, O2.y, s=500*np.sqrt(O2.m), c='magenta', marker='P')
    ax[0].set_title('x-y plane z='+str(z))
    
    '''
    add Lagrange points
    '''
    minima = (c==filters.minimum_filter(c, 3))  
        #Local minimum in x-y plane. Is stable Lagrange point if z=0.
    cx, cy = sphere().solve_L(O1, O2, z=z)
    ind_Lagrange = (np.abs(cy)<0.01)*(np.abs(cx)<0.01)
    minima_cy = (np.abs(cy)==filters.minimum_filter(np.abs(cy), footprint=np.ones((3, 1))))
    minima_cx = (np.abs(cx)==filters.minimum_filter(np.abs(cx), footprint=np.ones((1, 3))))
        #x(i, i0) is the same, so to find local minimum(near 0), should set footprint to search for a certain axis.
    ind_Lagrange = ind_Lagrange*minima_cy*minima_cx
    s_L = ax[0].scatter(x[ind_Lagrange], y[ind_Lagrange], s=100, c='g')  
        #All Lagrange points.
    s_stable = ax[0].scatter(x[minima], y[minima], s=200, marker='x', c='r')  
        #Only stable Lagrange points.
    legend = ax[0].legend([s_m, s_L, s_stable], ['Objects', 'All L points', 'x-y Stable'], 
                loc='upper left', prop={'size':26}, borderaxespad=0.)
    ax[0].add_artist(legend)
    
    '''
    x-z plane
    '''
    y=0
    x, z, c = sphere().solve_xz(O1, O2, y=y)
    contour = ax[1].contourf(x, z, c, np.arange(np.min(c), 10, 0.2), cmap=cm.jet)
        #Change np.arange(np.min(c), 20, 0.2) can give you better plot resolution.
    cb = fig.colorbar(contour, ax=ax[1])
    cb.set_label('Normalized Energy', fontsize=20)
    ax[1].set_aspect(1)
    ax[1].plot([-O1.range, O1.range], [0, 0], c='black')
    ax[1].plot([0, 0], [-O1.range, O1.range], c='black')
    s_m = ax[1].scatter(O1.x, O1.z, s=500*np.sqrt(O1.m), c='magenta', marker='P')
    ax[1].scatter(O2.x, O2.z, s=500*np.sqrt(O2.m), c='magenta', marker='P')
    ax[1].set_title('x-z plane y='+str(y))
    #fig.savefig('miu='+str(miu)+',z='+str(z)+'.jpg')
potential_plot(miu=0.1, z=0)
potential_plot(miu=0.1, z=1)
potential_plot(miu=0.5, z=0)